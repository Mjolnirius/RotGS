"""Prepare RotGS PNG sequences with automatic SAM 3.1 crops and product masks.

Uses a separate SAM 3.1 environment and checkpoint; see prepare_rotgs_sequence_SAM31.md.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import secrets
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

if __package__:
    from . import prepare_rotgs_sequence_SAM21 as shared
    from .rotgs_sam31_diagnostics import diagnostic_event, diagnostic_run, diagnostic_save_frame, diagnostic_sources, diagnostic_span
else:
    import prepare_rotgs_sequence_SAM21 as shared
    from rotgs_sam31_diagnostics import diagnostic_event, diagnostic_run, diagnostic_save_frame, diagnostic_sources, diagnostic_span

legacy = shared.legacy
validate_prompts = shared.validate_prompts
propagate_masks = shared.propagate_masks


class SAM31Segmenter:
    """Detect independent rotation views, with optional per-anchor corrections."""

    def __init__(self, frames_folder, checkpoint, device="cuda", threshold=0.5,
                 text_prompt=None):
        if device != "cuda":
            raise ValueError("the official SAM 3.1 video builder requires CUDA")
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("mask probability threshold must be between 0 and 1")
        try:
            import torch
            from sam3.model_builder import build_sam3_predictor
        except ImportError as exc:
            raise RuntimeError("Install current SAM 3.1; see prepare_rotgs_sequence_SAM31.md") from exc
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("SAM 3.1 requires a CUDA GPU supporting bfloat16")
        self.torch = torch
        self.threshold = threshold
        self.text_prompt = text_prompt
        self.session_id = None
        self.object_id = None
        self.frames_folder = Path(frames_folder)
        with Image.open(self.frames_folder / "0000.jpg") as frame:
            self.size = frame.size
        diagnostic_event("backend", torch_version=getattr(torch, "__version__", None),
                         cuda_build=getattr(getattr(torch, "version", None), "cuda", None),
                         working_size=list(self.size))
        with diagnostic_span("model_build", checkpoint=str(checkpoint), torch=torch):
            self.predictor = build_sam3_predictor(
                checkpoint_path=str(checkpoint), version="sam3.1",
                use_fa3=False, compile=False, warm_up=False,
                async_loading_frames=False, default_output_prob_thresh=threshold,
            )
        # Bound batching for any upstream path using it. Current interactive
        # add_prompt disables batched grounding; log requests to distinguish it
        # from video propagation. Postprocessing still uses its configured batch.
        self.predictor.model.batched_grounding_batch_size = 1
        self.predictor.model.postprocess_batch_size = 1
        diagnostic_sources(build_sam3_predictor, type(self.predictor.model))
        diagnostic_event("model_configuration", model_class=type(self.predictor.model).__name__,
                         grounding_batch_limit=1, postprocess_batch_size=1,
                         use_batched_grounding=getattr(self.predictor.model, "use_batched_grounding", None))
        # Some upstream releases pass this SAM 2-only option unconditionally
        # from start_session to a multiplex init_state that does not accept it.
        init_state = self.predictor.model.init_state
        parameters = inspect.signature(init_state).parameters
        if "offload_state_to_cpu" not in parameters and not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            def compatible_init_state(**kwargs):
                kwargs.pop("offload_state_to_cpu", None)
                return init_state(**kwargs)
            self.predictor.model.init_state = compatible_init_state
        try:
            with diagnostic_span("start_session", torch=torch, frames_folder=str(frames_folder)):
                response = self.predictor.handle_request(dict(
                    type="start_session", resource_path=str(frames_folder),
                    offload_video_to_cpu=True,
                ))
            self.session_id = response["session_id"]
        except BaseException:
            self.close()
            raise

    def context(self):
        return self.torch.inference_mode()

    def _request(self, request_type, **kwargs):
        index = kwargs.get("frame_index")
        folder = getattr(self, "frames_folder", None)
        if index is not None and folder is not None:
            diagnostic_save_frame(folder / f"{index:04d}.jpg", index)
        with diagnostic_span(request_type, torch=self.torch, **kwargs):
            return self.predictor.handle_request(dict(
                type=request_type, session_id=self.session_id, **kwargs,
            ))

    def _outputs(self, response):
        index = int(response["frame_index"])
        outputs = response["outputs"]
        ids = np.asarray(outputs["out_obj_ids"])
        masks = np.asarray(outputs["out_binary_masks"])
        if ids.ndim != 1 or masks.shape != (len(ids), self.size[1], self.size[0]):
            raise RuntimeError(f"frame {index}: unexpected SAM 3.1 output dimensions")
        if not np.isfinite(masks).all():
            raise RuntimeError(f"frame {index}: non-finite SAM 3.1 output")
        diagnostic_event("sam_output", frame_index=index, object_ids=ids.tolist(),
                         mask_shape=list(masks.shape), foreground_pixels=[int(mask.sum()) for mask in masks])
        return index, ids, masks.astype(bool)

    def _select_object(self, response, prompt):
        index, ids, masks = self._outputs(response)
        if not len(ids):
            raise ValueError(f"frame {index}: no product detected; correct box/text prompt")
        # Prefer masks covering positive clicks, then matching the product box.
        scores = []
        for mask in masks:
            positives = sum(bool(mask[int(y), int(x)]) for (x, y), label in
                            zip(prompt["points"], prompt["labels"]) if label == 1)
            negatives = sum(bool(mask[int(y), int(x)]) for (x, y), label in
                            zip(prompt["points"], prompt["labels"]) if label == 0)
            if prompt["box"]:
                x0, y0, x1, y1 = prompt["box"]
                region = np.zeros_like(mask)
                region[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
                overlap = np.count_nonzero(mask & region) / max(np.count_nonzero(mask | region), 1)
            else:
                overlap = np.count_nonzero(mask) / mask.size
            scores.append((positives, -negatives, overlap))
        return int(ids[max(range(len(ids)), key=lambda i: scores[i])])

    def _mask(self, response, prompt=None, *, preserve_components=False):
        index, ids, masks = self._outputs(response)
        positions = np.flatnonzero(ids == self.object_id)
        if len(positions) != 1:
            raise ValueError(f"frame {index}: tracked product missing; add/correct an anchor")
        try:
            mask = masks[positions[0]]
            if preserve_components:
                if not mask.any():
                    raise ValueError("SAM produced an empty product mask")
                return index, mask
            return index, shared.product_component(mask, prompt)
        except ValueError as exc:
            raise ValueError(f"frame {index}: {exc}") from exc

    def _refine(self, prompt):
        return self._request(
            "add_prompt", frame_index=prompt["frame_index"], obj_id=self.object_id,
            points=prompt["points"], point_labels=prompt["labels"],
            rel_coordinates=False, clear_old_points=True,
            output_prob_thresh=self.threshold,
        )

    def _seed(self, prompt, *, refine=True):
        self._request("reset_session")
        self.object_id = None
        index = prompt["frame_index"]
        if prompt["box"] or self.text_prompt or not refine:
            request = dict(frame_index=index, output_prob_thresh=self.threshold)
            if self.text_prompt:
                request["text"] = self.text_prompt
            elif not prompt["box"]:
                request["text"] = "object"
            if prompt["box"]:
                x0, y0, x1, y1 = prompt["box"]
                w, h = self.size
                request.update(bounding_boxes=[[x0 / w, y0 / h, (x1-x0) / w, (y1-y0) / h]],
                               bounding_box_labels=[1])
            response = self._request("add_prompt", **request)
            self.object_id = self._select_object(response, prompt)
            _, ids, _ = self._outputs(response)
            for object_id in ids:
                if int(object_id) != self.object_id:
                    self._request("remove_object", frame_index=index, obj_id=int(object_id))
        if prompt["points"] and refine:
            if self.object_id is None:
                self.object_id = 1
            # Instance refinement accepts points alone, without a text/box prompt.
            response = self._refine(prompt)
        if self.object_id is None:
            raise ValueError("an anchor requires a box or positive points")
        return response

    def preview(self, prompt):
        with self.context():
            return self._mask(self._seed(prompt), prompt)[1]

    def detect(self, index, product_box=None, *, preserve_components=False):
        """Detect independently; use box fallbacks only when a crop is known."""
        with self.context():
            prompt = dict(frame_index=index, box=product_box, points=[], labels=[])
            w, h = self.size
            attempts = []
            if self.text_prompt:
                attempts.append(("text", dict(text=self.text_prompt)))
                fallback = self.text_prompt.split()[-1]
                if fallback != self.text_prompt:
                    attempts.append(("text_fallback", dict(text=fallback)))
            if product_box is not None:
                x0, y0, x1, y1 = product_box
                box_request = dict(
                    bounding_boxes=[[x0/w, y0/h, (x1-x0)/w, (y1-y0)/h]],
                    bounding_box_labels=[1],
                )
                if self.text_prompt:
                    attempts.append(("text_and_box", dict(text=self.text_prompt, **box_request)))
                attempts.append(("box", box_request))
            for method, request in attempts:
                diagnostic_event("detection_attempt", frame_index=index, method=method,
                                 prompt=request.get("text"), box=product_box)
                self._request("reset_session")
                response = self._request("add_prompt", frame_index=index,
                                         output_prob_thresh=self.threshold, **request)
                _, ids, masks = self._outputs(response)
                if not len(ids):
                    diagnostic_event("detection_rejected", reason="no objects")
                    continue
                # A product crop includes a margin. Reject background/complement
                # masks occupying most of its perimeter, even when their area
                # gives a higher box IoU than the actual product.
                perimeter = np.concatenate((masks[:, 0], masks[:, -1],
                                            masks[:, :, 0], masks[:, :, -1]), axis=1)
                eligible = perimeter.mean(axis=1) < 0.5
                if not eligible.any():
                    diagnostic_event("detection_rejected", reason="background perimeter",
                                     perimeter_occupancy=perimeter.mean(axis=1).tolist())
                    continue
                response = dict(response, outputs=dict(response["outputs"],
                    out_obj_ids=ids[eligible], out_binary_masks=masks[eligible]))
                self.object_id = self._select_object(response, prompt)
                self.last_detection_method = method
                self.last_detection_prompt = request.get("text")
                diagnostic_event("detection_selected", object_id=self.object_id, method=method)
                return self._mask(response, preserve_components=preserve_components)[1]
            if product_box is None:
                raise ValueError(f"frame {index}: automatic crop detection failed; correct --text-prompt or use --manual_crop")
            raise ValueError(f"frame {index}: text and crop-box detection both failed; inspect crop/product prompt")

    def track(self, prompt, end, reverse=False):
        """Evaluate each view independently; clicks apply only to their anchor."""
        anchor = prompt["frame_index"]
        step = -1 if reverse else 1
        box = prompt["box"] or [0, 0, *self.size]
        for index in range(anchor, end + step, step):
            if index == anchor:
                mask = self.preview(prompt)
            else:
                mask = self.detect(index, box)
            yield index, mask

    def close(self):
        if self.predictor is None:
            return
        original_error = sys.exc_info()[1]
        errors = []

        def cleanup(operation, function):
            try:
                with diagnostic_span(operation):
                    function()
            except Exception as exc:
                errors.append(exc)
                diagnostic_event("cleanup_error", operation=operation, error=str(exc))

        if self.session_id is not None:
            cleanup("close_session_cleanup", lambda: self._request("close_session"))
        self.session_id = None
        cleanup("predictor_shutdown", self.predictor.shutdown)
        # The multiplex tracker enters autocast before the video predictor.
        # Exit both in reverse order; leaving the tracker context active keeps
        # cached cast weights alive when a second crop-segmentation model loads.
        tracker = getattr(getattr(self.predictor, "model", None), "tracker", None)
        for owner, operation in ((self.predictor, "autocast_exit"),
                                 (tracker, "tracker_autocast_exit")):
            context = getattr(owner, "bf16_context", None)
            if context is not None:
                cleanup(operation, lambda: context.__exit__(None, None, None))
                owner.bf16_context = None
        # Do not retain the tracker locally while collecting reference cycles.
        tracker = owner = context = None
        # Release the model and the compatibility closure's reference cycle.
        self.predictor = None
        cleanup("garbage_collection", gc.collect)
        cleanup("cuda_empty_cache", self.torch.cuda.empty_cache)
        if errors:
            if original_error is None:
                raise errors[0]
            if hasattr(original_error, "add_note"):
                for error in errors:
                    original_error.add_note(f"SAM cleanup also failed: {error}")


def detect_masks(segmenter, frame_count, box, folder, *, source_views=None):
    """Automatic concept detection for each rotation view, without manual clicks."""
    folder.mkdir()
    rows = []
    for index in tqdm(range(frame_count), desc="Detecting product with SAM 3.1"):
        item = source_views[index] if source_views is not None else None
        diagnostic_event("frame_begin", frame_index=index,
                         source_image=str(item.path) if item is not None else None,
                         angle_degrees=item.angle if item is not None else None,
                         method=None, prompt=None)
        mask = segmenter.detect(index, box, preserve_components=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{index:04d}.png")
        diagnostic_event("frame_end", frame_index=index)
        rows.append(dict(frame_index=index, anchor=None, distance=0, min_overlap_iou=None,
                         detection_method=segmenter.last_detection_method,
                         detection_prompt=getattr(segmenter, "last_detection_prompt", None)))
    return rows


def _mask_box_in_source(mask, source_size):
    """Map exclusive mask bounds back to native pixels, rounding outward."""
    if mask.ndim != 2 or not mask.any():
        raise ValueError("automatic crop detection produced an empty or invalid mask")
    ys, xs = np.nonzero(mask)
    width, height = source_size
    mask_height, mask_width = mask.shape
    # Integer arithmetic avoids floating-point errors at exact pixel boundaries.
    return (
        int(xs.min()) * width // mask_width,
        int(ys.min()) * height // mask_height,
        ((int(xs.max()) + 1) * width + mask_width - 1) // mask_width,
        ((int(ys.max()) + 1) * height + mask_height - 1) // mask_height,
    )


def _discover_crop(ordered, undistortion, source_size, checkpoint, device,
                   mask_threshold, text_prompt, segmentation_max_width, margin):
    """Find one native square covering the detected product in every view."""
    width, height = source_size
    detection_width = min(segmentation_max_width, width)
    detection_size = (detection_width, max(1, round(height * detection_width / width)))
    rows = []
    diagnostic_event("phase", phase="crop_prepare", working_size=list(detection_size))
    with tempfile.TemporaryDirectory(prefix="rotgs-sam31-crop-") as directory:
        frames_folder = Path(directory)
        for index, item in enumerate(tqdm(ordered, desc="Preparing automatic crop views")):
            diagnostic_event("frame_prepare", frame_index=index, source_image=str(item.path),
                             angle_degrees=item.angle, method=None, prompt=None)
            with Image.open(item.path) as image:
                if image.format != "PNG" or image.size != source_size:
                    raise ValueError(f"inconsistent source format/size: {item.path}")
                corrected = legacy._remap_rgb(image, undistortion)
            with corrected, corrected.resize(detection_size, Image.Resampling.LANCZOS) as working:
                working.save(frames_folder / f"{index:04d}.jpg", quality=100, subsampling=0)
        diagnostic_event("phase", phase="crop_inference")
        segmenter = SAM31Segmenter(frames_folder, checkpoint, device, mask_threshold, text_prompt)
        try:
            for index, item in enumerate(tqdm(ordered, desc="Finding crop with SAM 3.1")):
                diagnostic_event("frame_begin", frame_index=index, source_image=str(item.path),
                                 angle_degrees=item.angle, method=None, prompt=None)
                try:
                    mask = segmenter.detect(index, preserve_components=True)
                    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                        raise ValueError("detected product touches the full image boundary")
                    box = _mask_box_in_source(mask, source_size)
                except ValueError as exc:
                    raise ValueError(
                        f"automatic crop, frame {index} ({item.path.name}): {exc}; "
                        "check --text-prompt or use --manual_crop"
                    ) from exc
                rows.append(dict(
                    frame_index=index, image=item.path.name, angle_degrees=item.angle,
                    box_left_top_right_bottom=list(box),
                    detection_method=segmenter.last_detection_method,
                    detection_prompt=getattr(segmenter, "last_detection_prompt", None),
                ))
                diagnostic_event("frame_end", frame_index=index, box=list(box))
        finally:
            segmenter.close()
    selection = (
        min(row["box_left_top_right_bottom"][0] for row in rows),
        min(row["box_left_top_right_bottom"][1] for row in rows),
        max(row["box_left_top_right_bottom"][2] for row in rows),
        max(row["box_left_top_right_bottom"][3] for row in rows),
    )
    expanded = (selection[0] - margin, selection[1] - margin,
                selection[2] + margin, selection[3] + margin)
    try:
        crop = legacy.crop_box_for_selection(expanded, source_size, True)
        legacy._validate_valid_crop(crop, undistortion)
    except ValueError as exc:
        raise ValueError(
            f"automatic square crop with {margin}px margin does not fit the valid image: {exc}; "
            "reduce --crop-margin or use --manual_crop"
        ) from exc
    discovery = dict(
        working_size=list(detection_size), coordinate_space="full_resolution_undistorted",
        margin_pixels=margin, union_box_left_top_right_bottom=list(selection),
        crop_box_left_top_right_bottom=list(crop), frames=rows,
    )
    print(f"SAM31AC: product bounds {selection}; square crop {crop}; margin {margin}px", flush=True)
    return selection, crop, discovery


def _save_crop_preview(image, row, crop, path):
    """Draw the detected box and shared crop over the full undistorted view."""
    with image.copy() as preview:
        preview.thumbnail((480, 480))
        sx, sy = preview.width / image.width, preview.height / image.height
        draw = ImageDraw.Draw(preview)
        for box, color in ((crop, "lime"), (row["box_left_top_right_bottom"], "magenta")):
            left, top, right, bottom = box
            draw.rectangle((left * sx, top * sy, (right - 1) * sx, (bottom - 1) * sy),
                           outline=color, width=2)
        preview.save(path)


def _write_crop_review(folder, discovery, ordered):
    import html

    cards = []
    for row, item in zip(discovery["frames"], ordered):
        cards.append(
            f'<figure><a href="{item.output_name}"><img src="{item.output_name}"></a>'
            f'<figcaption>{html.escape(item.path.name)} / {item.angle}°<br>'
            f'{html.escape(str(row["box_left_top_right_bottom"]))}<br>'
            f'{html.escape(row["detection_method"])}: '
            f'{html.escape(row["detection_prompt"] or "")}</figcaption></figure>'
        )
    (folder / "detections.json").write_text(json.dumps(discovery, indent=2), encoding="utf-8")
    (folder / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>SAM31AC crop review</title>'
        '<style>body{font:16px system-ui;background:#20242b;color:white}'
        'main{display:flex;flex-wrap:wrap}figure{margin:12px}img{max-width:360px}</style>'
        '<h1>Automatic crop review</h1><p>Magenta: detected product box. '
        f'Green: shared square crop, with {discovery["margin_pixels"]} native pixels of margin per side. '
        '<a href="detections.json">Detection coordinates</a> · '
        '<a href="../mask_review/index.html">Final mask review</a></p><main>'
        + "".join(cards) + "</main>", encoding="utf-8",
    )


@diagnostic_run
def prepare_rotgs_sequence_SAM31(
    folder_name, cameras_file, *, checkpoint, device="cuda", text_prompt=None,
    target_width=None, calibration_file=None, selection=None, destination_folder=None,
    prompts_json=None, box_only=False, interactive=False, anchor_interval=18, segmentation_max_width=1600,
    mask_threshold=0.5, web_port=8765, angle_token_from_right=None,
    aruco_side_length_mm=40.0, skip_tae=False, manual_crop=False, crop_margin=10,
    diagnostics_dir=None, cuda_sync=False,
):
    """Undistort/crop PNGs, detect the product, and publish a separate RGBA dataset.

    Selection uses full-resolution undistorted pixels. Saved SAM prompts use
    cropped working-resolution pixels, recorded explicitly in the JSON file.
    Source alpha is discarded, not used as an object-membership mask.
    """
    if device != "cuda":
        raise ValueError("the official SAM 3.1 video builder requires CUDA")
    if not 0 < mask_threshold < 1:
        raise ValueError("mask probability threshold must be between 0 and 1")
    if anchor_interval < 1 or segmentation_max_width < 1 or not np.isfinite(mask_threshold):
        raise ValueError("interval/working width must be positive; threshold must be finite")
    if target_width is not None and target_width <= 0:
        raise ValueError("target width must be positive")
    if not 1 <= web_port <= 65535:
        raise ValueError("web port must be between 1 and 65535")
    if type(crop_margin) is not int or crop_margin < 0:
        raise ValueError("crop margin must be a non-negative integer in native pixels")
    if box_only and prompts_json:
        raise ValueError("choose either --box-only or --prompts-json")
    if interactive and (box_only or prompts_json):
        raise ValueError("--interactive cannot be combined with --box-only or --prompts-json")
    automatic = not (interactive or box_only or prompts_json)
    automatic_crop = selection is None and not manual_crop
    if (automatic or automatic_crop) and not (isinstance(text_prompt, str) and text_prompt.strip()):
        raise ValueError(
            "automatic crop/segmentation requires --text-prompt; "
            "use --manual_crop with --interactive for manual crop and anchors"
        )
    if aruco_side_length_mm is not None and (
        not np.isfinite(aruco_side_length_mm) or aruco_side_length_mm <= 0
    ):
        raise ValueError("ArUco side length must be positive and finite")
    source = Path(folder_name).expanduser().resolve(strict=True)
    cameras = Path(cameras_file).expanduser().resolve(strict=True)
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("checkpoint must be a local SAM 3.1 checkpoint file")
    # Fail before an interactive selection if the optional dependency is missing.
    import importlib.util
    if importlib.util.find_spec("sam3") is None:
        raise RuntimeError("SAM 3.1 is not installed; see prepare_rotgs_sequence_SAM31.md")
    ordered, angle_index = legacy.detect_angle_order(
        sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
        angle_token_from_right=angle_token_from_right,
    )
    with Image.open(ordered[0].path) as image:
        source_size = image.size
    diagnostic_event("sequence", phase="input", source_size=list(source_size),
                     frames=[dict(frame_index=i, source_image=str(item.path), angle_degrees=item.angle)
                             for i, item in enumerate(ordered)])
    camera_text = cameras.read_text(encoding="utf-8")
    undistortion = legacy._resolve_undistortion(calibration_file, cameras, source_size, camera_text)
    crop_discovery = None
    if automatic_crop:
        selection, crop, crop_discovery = _discover_crop(
            ordered, undistortion, source_size, checkpoint, device, mask_threshold,
            text_prompt, segmentation_max_width, crop_margin,
        )
        crop_selection_method = "sam31_automatic"
    elif selection is None:
        with Image.open(ordered[0].path) as image:
            first = legacy._remap_rgb(image, undistortion)
        with first:
            print("Select a crop containing the product at EVERY rotation angle, with a small margin.", flush=True)
            selection, crop, _, target_width = shared._choose_crop(
                first, ordered, undistortion, segmentation_max_width, target_width, web_port, model_label="SAM 3.1",
            )
        crop_selection_method = "manual_six_view"
    else:
        selection = tuple(selection)
        crop = legacy.crop_box_for_selection(selection, source_size, True)
        crop_selection_method = "provided"
    legacy._validate_valid_crop(crop, undistortion)
    crop_size = (crop[2] - crop[0], crop[3] - crop[1])
    output_size = legacy.calculate_output_size(crop_size, target_width or crop_size[0])
    working_size = legacy.calculate_output_size(crop_size, min(segmentation_max_width, crop_size[0]))
    output_suffix = "SAM31AC" if automatic_crop else "und_rn_roi_sqr_SAM31"
    output = (Path(destination_folder).expanduser().resolve() if destination_folder else
              source.with_name(f"{source.name}_{output_suffix}_{output_size[0]}"))
    if output == source or source in output.parents:
        raise ValueError("output must be separate from the source folder")
    if output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent does not exist: {output.parent}")
    relative_box = legacy._relative_product_box(selection, crop)
    sx, sy = working_size[0] / crop_size[0], working_size[1] / crop_size[1]
    box = [relative_box[0] * sx, relative_box[1] * sy, relative_box[2] * sx, relative_box[3] * sy]
    anchors = sorted(set(range(0, len(ordered), anchor_interval)) | {len(ordered) - 1})
    document = dict(
        version=1, text_prompt=text_prompt, mode="automatic" if automatic else "anchors",
        coordinate_size=list(working_size),
        crop_box_left_top_right_bottom=list(crop), source_size=list(source_size),
        crop_selection_method=crop_selection_method,
        images=[item.path.name for item in ordered],
        frames=[dict(frame_index=i, box=box.copy(), points=[], labels=[]) for i in anchors],
    )
    if prompts_json:
        supplied = json.loads(Path(prompts_json).read_text(encoding="utf-8"))
        for key in ("images", "crop_box_left_top_right_bottom", "source_size"):
            if supplied.get(key) != document[key]:
                raise ValueError(f"saved prompts do not match this sequence/crop: {key}")
        document["frames"] = validate_prompts(supplied, len(ordered), working_size)

    with tempfile.TemporaryDirectory(prefix="rotgs-sam31-work-") as work_dir, \
         tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as staging_dir:
        work, staging = Path(work_dir), Path(staging_dir)
        frames_folder = work / "frames"
        frames_folder.mkdir()
        # Lossless native-resolution RGB is retained for final RGBA export.
        rgb_folder = work / "rgb"
        rgb_folder.mkdir()
        if crop_discovery is not None:
            crop_review = staging / "crop_review"
            crop_review.mkdir()
        for index, item in enumerate(tqdm(ordered, desc="Undistorting source PNGs")):
            diagnostic_event("frame_prepare", phase="native_rgb", frame_index=index,
                             source_image=str(item.path), angle_degrees=item.angle)
            with Image.open(item.path) as image:
                if image.format != "PNG" or image.size != source_size:
                    raise ValueError(f"inconsistent source format/size: {item.path}")
                corrected = legacy._remap_rgb(image, undistortion)
            with corrected:
                if crop_discovery is not None:
                    _save_crop_preview(corrected, crop_discovery["frames"][index], crop,
                                       crop_review / item.output_name)
                with corrected.crop(crop) as cropped:
                    cropped.save(rgb_folder / item.output_name)
                    with cropped.resize(working_size, Image.Resampling.LANCZOS) as working:
                        # Run final segmentation on the native RGB crop, so the
                        # object occupies more of SAM's fixed internal resolution.
                        # Only temporary inference copies use JPEG, never exported RGB.
                        working.save(frames_folder / f"{index:04d}.jpg", quality=100, subsampling=0)
        if crop_discovery is not None:
            _write_crop_review(crop_review, crop_discovery, ordered)
        tae_enabled = not skip_tae and aruco_side_length_mm is not None
        if tae_enabled:
            with diagnostic_span("turntable_angle_estimation", phase="tae"):
                legacy._run_turntable_angle_estimation(
                    source, undistortion.calibration_file, staging, aruco_side_length_mm,
                )
        diagnostic_event("phase", phase="crop_segmentation", working_size=list(working_size))
        segmenter = SAM31Segmenter(frames_folder, checkpoint, device, mask_threshold, text_prompt)
        try:
            if interactive:
                document["frames"] = shared.prompt_browser(segmenter, frames_folder, document, web_port, model_label="SAM 3.1")
            prompts = validate_prompts(document, len(ordered), working_size)
            document["frames"] = prompts
            # Keep prompts available even if segmentation or dataset publication fails.
            recovery_file = output.with_name(f"{output.name}.prompts-{secrets.token_hex(4)}.json")
            with recovery_file.open("x", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
            print(f"Saved prompts (also retained on failure): {recovery_file}", flush=True)
            if automatic:
                rows = detect_masks(segmenter, len(ordered), box, work / "masks", source_views=ordered)
            else:
                rows = propagate_masks(segmenter, prompts, len(ordered), work / "masks")
        finally:
            segmenter.close()
        images_folder = staging / "images"
        sparse_folder = staging / "sparse" / "0"
        images_folder.mkdir()
        sparse_folder.mkdir(parents=True)
        counts = dict(pixels=0, alpha_eq_0=0, alpha_eq_255=0, alpha_between_0_and_255=0)
        per_image = []
        for index, item in enumerate(tqdm(ordered, desc="Writing product RGBA PNGs")):
            diagnostic_event("frame_export", phase="rgba_export", frame_index=index,
                             source_image=str(item.path), angle_degrees=item.angle)
            with Image.open(work / "masks" / f"{index:04d}.png") as alpha, \
                 Image.open(rgb_folder / item.output_name) as rgb:
                mask = np.asarray(alpha) > 0
                rows[index]["touches_crop_edge"] = bool(
                    mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
                )
                # Binary membership at native crop size; only the final downsample
                # creates fractional edge coverage. RGB is never generated/inpainted.
                rgba = rgb.convert("RGBA")
                with alpha.resize(crop_size, Image.Resampling.NEAREST) as native_alpha:
                    rgba.putalpha(native_alpha)
                with rgba:
                    with rgba.resize(output_size, Image.Resampling.LANCZOS) as resized:
                        resized.save(images_folder / item.output_name)
                        item_counts = legacy._alpha_counts(np.asarray(resized.getchannel("A")))
                for key in counts:
                    counts[key] += item_counts[key]
                per_image.append(dict(image=item.output_name, **legacy._with_percentages(item_counts)))
        shared._write_review(staging, rows, ordered, model_label="SAM 3.1", prompts_filename="sam31_prompts.json")
        calibrated_text = legacy._cameras_text_with_matrix(
            camera_text, source_size, undistortion.output_camera_matrix,
        )
        transformed = legacy.transform_camera_intrinsics(calibrated_text, source_size, crop, output_size)
        (sparse_folder / "cameras.txt").write_text(transformed, encoding="utf-8")
        metadata = dict(
            input_folder=str(source), input_cameras_file=str(cameras),
            input_cameras_sha256=hashlib.sha256(camera_text.encode()).hexdigest(),
            source_format="PNG", source_size=list(source_size), image_count=len(ordered),
            alpha_mode="sam31_product", overwrite_alpha_mask=True,
            crop_box_left_top_right_bottom=list(crop), crop_size=list(crop_size),
            product_selection_left_top_right_bottom=list(selection), crop_mode="smallest_square",
            crop_selection_method=crop_selection_method, automatic_crop=crop_discovery,
            output_size=list(output_size), angles_degrees=[item.angle for item in ordered],
            angle_token_from_right=angle_index,
            renamed_images=[dict(source=item.path.name, angle_degrees=item.angle, output=item.output_name)
                            for item in ordered],
            undistortion=dict(applied=True, calibration_file=str(undistortion.calibration_file),
                              calibration_sha256=undistortion.calibration_sha256,
                              output_camera_matrix=undistortion.output_camera_matrix.tolist(),
                              alpha_remap="discarded_then_regenerated_after_undistortion"),
            segmentation=dict(backend="sam3.1", model="multiplex", checkpoint=str(checkpoint),
                              text_prompt=text_prompt, threshold_kind="probability",
                              working_size=list(working_size),
                              mask_threshold=mask_threshold, reused_crop_detection_masks=False,
                              input_view="cropped", inference_passes=2 if automatic_crop else 1,
                              inference_batch_size=1, postprocess_batch_size=1,
                              mode="per_frame_concept" if automatic else "anchor_guided_detection",
                              merge="none" if automatic else "nearest_anchor",
                              independent_forward_reverse=not automatic,
                              prompts_reviewed_in_browser=interactive,
                              source_alpha_used=False),
            turntable_angle_estimation=dict(
                status="completed" if tae_enabled else "skipped",
                marker_size_mm=aruco_side_length_mm,
                input_folder=str(source),
                calibration_file=str(undistortion.calibration_file),
                output_folder=legacy.TURNTABLE_ANGLE_OUTPUT_NAME if tae_enabled else None,
            ),
            alpha_statistics=legacy._with_percentages(counts), per_image_alpha_statistics=per_image,
            mask_review=rows,
        )
        (staging / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (staging / "sam31_prompts.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        # Include recovery prompts before publishing; remove the external copy only on success.
        (staging / recovery_file.name).write_bytes(recovery_file.read_bytes())
        # Publish only after all images, metadata, masks and optional angles succeed.
        if output.exists():
            raise FileExistsError(f"output appeared during processing: {output}")
        staging.rename(output)
        recovery_file.unlink()
    print(f"Prepared {len(ordered)} frames: {output}")
    print(f"Recovery prompts: {output / recovery_file.name}")
    print(f"Inspect masks: {output / 'mask_review' / 'index.html'}")
    if crop_discovery is not None:
        print(f"Inspect automatic crop: {output / 'crop_review' / 'index.html'}")
    return dict(output_folder=str(output), processed=len(ordered), metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder_name", help="original angle-named PNG sequence")
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path, help="local SAM 3.1 weights")
    parser.add_argument("--text-prompt", help="concept prompt required for automatic crop/masks, e.g. beverage can")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--calibration", type=Path, dest="calibration_file")
    parser.add_argument("--width", type=int, dest="target_width")
    parser.add_argument("--selection", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="replay crop selection in full-resolution UNDISTORTED pixels; overrides automatic crop")
    parser.add_argument("--manual_crop", "--manual-crop", action="store_true",
                        help="draw a crop and review six angles in the browser instead of automatic SAM31AC crop")
    parser.add_argument("--crop-margin", type=int, default=10,
                        help="automatic square crop margin per side in native undistorted pixels (default: 10)")
    parser.add_argument("--output", type=Path, dest="destination_folder")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--interactive", action="store_true", help="opt in to manual box/click correction on selected views")
    prompt_group.add_argument("--prompts-json", type=Path, help="replay/edit saved sam31_prompts.json")
    prompt_group.add_argument("--box-only", action="store_true", help="skip prompt editor; seed all anchors with crop selection box")
    parser.add_argument("--anchor-interval", type=int, default=18, help="initial anchor spacing in frames (18 = 90 degrees)")
    parser.add_argument("--segmentation-max-width", type=int, default=1600)
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="SAM mask probability threshold (0 < threshold < 1)")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--angle-token-from-right", type=int)
    parser.add_argument("--aruco-side-length-mm", type=float, default=40.0,
                        help="black ArUco square side length in mm for TAE (default: 40)")
    parser.add_argument("--skip-tae", action="store_true",
                        help="skip the default turntable angle estimation")
    parser.add_argument("--diagnostics-dir", type=Path,
                        help="save console, frame/request events, telemetry, failing input and traceback in a new run folder")
    parser.add_argument("--cuda-sync", action="store_true",
                        help="diagnose CUDA failures with synchronous launches and request synchronization; slower")
    args = vars(parser.parse_args())
    args["cameras_file"] = args.pop("cameras")
    try:
        prepare_rotgs_sequence_SAM31(**args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyboardInterrupt) as exc:
        print(f"Error: {exc or 'cancelled'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
