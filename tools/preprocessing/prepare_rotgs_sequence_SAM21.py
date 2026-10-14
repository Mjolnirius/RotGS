"""Prepare product-only RotGS PNG sequences with prompted SAM 2.1 tracking.

The Lab-based pipeline is intentionally unchanged. See prepare_rotgs_sequence_SAM21.md.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import secrets
import sys
import tempfile
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image
from tqdm import tqdm

if __package__:
    from . import prepare_rotgs_sequence_png as legacy
else:
    import prepare_rotgs_sequence_png as legacy


CONFIGS = {
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
}


def validate_prompts(document, frame_count, size):
    """Validate prompts in cropped working-image pixel coordinates."""
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("prompts JSON must have version: 1")
    if document.get("coordinate_size") != list(size):
        raise ValueError(f"prompt coordinate_size must be {list(size)}")
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("at least one prompted frame is required")
    result, seen = [], set()
    w, h = size
    for frame in frames:
        index = frame.get("frame_index")
        if type(index) is not int or not 0 <= index < frame_count or index in seen:
            raise ValueError("frame_index must be unique and within the sequence")
        seen.add(index)
        box = frame.get("box")
        if box is not None:
            box = np.asarray(box, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("box must contain four finite coordinates")
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError("box lies outside the working image or has no area")
            box = box.tolist()
        points = np.asarray(frame.get("points", []), dtype=float)
        labels = frame.get("labels", [])
        if points.size == 0:
            points = np.empty((0, 2))
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("points must be finite [x, y] coordinates")
        if not isinstance(labels, list) or len(labels) != len(points) or any(
            type(label) is not int or label not in (0, 1) for label in labels
        ):
            raise ValueError("each point needs label 1 (product) or 0 (background)")
        if len(points) and not (
            (points >= 0).all() and (points[:, 0] < w).all() and (points[:, 1] < h).all()
        ):
            raise ValueError("point lies outside the working image")
        if box is None and 1 not in labels:
            raise ValueError("each anchor needs a box or a positive product point")
        result.append(dict(frame_index=index, box=box, points=points.tolist(), labels=labels))
    return sorted(result, key=lambda frame: frame["frame_index"])


def product_component(mask, prompt=None):
    """Keep components seeded by positive clicks; otherwise keep the largest.

    No hole filling or erosion: internal openings and thin edges are preserved.
    Anchor coordinates are never reused on another frame of a rotating product.
    """
    import cv2

    count, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count == 1:
        raise ValueError("SAM produced an empty product mask; correct the anchor prompts")
    selected = set()
    if prompt:
        for (x, y), label in zip(prompt["points"], prompt["labels"]):
            if label == 1 and components[int(y), int(x)] != 0:
                selected.add(int(components[int(y), int(x)]))
    if not selected:
        selected.add(1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return np.isin(components, list(selected))


class SAM21Segmenter:
    def __init__(self, frames_folder, checkpoint, model, device, threshold=0.0):
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2.1 is not installed. See tools/preprocessing/prepare_rotgs_sequence_SAM21.md "
                "for installation in a separate environment."
            ) from exc
        self.torch = torch
        self.device = device
        self.threshold = threshold
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for a slower run")
        with self.context():
            self.predictor = build_sam2_video_predictor(
                CONFIGS[model], str(checkpoint), device=device,
                # Avoid automatic hole filling and optional CUDA postprocessing.
                apply_postprocessing=False,
            )
            self.state = self.predictor.init_state(
                video_path=str(frames_folder), offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )

    def context(self):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(self.torch.inference_mode())
        if self.device == "cuda":
            dtype = (self.torch.bfloat16 if self.torch.cuda.is_bf16_supported()
                     else self.torch.float16)
            stack.enter_context(self.torch.autocast("cuda", dtype=dtype))
        return stack

    def _seed(self, prompt):
        self.predictor.reset_state(self.state)
        return self.predictor.add_new_points_or_box(
            inference_state=self.state, frame_idx=prompt["frame_index"], obj_id=1,
            box=np.asarray(prompt["box"], dtype=np.float32) if prompt["box"] else None,
            points=np.asarray(prompt["points"], dtype=np.float32) if prompt["points"] else None,
            labels=np.asarray(prompt["labels"], dtype=np.int32) if prompt["points"] else None,
            clear_old_points=True,
        )

    def _mask(self, result, prompt=None):
        index, ids, logits = result
        if ids != [1]:
            raise RuntimeError("expected exactly one tracked product")
        values = logits[0, 0].float().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite SAM output on frame {index}")
        try:
            return int(index), product_component(values > self.threshold, prompt)
        except ValueError as exc:
            raise ValueError(f"frame {index}: {exc}") from exc

    def preview(self, prompt):
        with self.context():
            return self._mask(self._seed(prompt), prompt)[1]

    def track(self, prompt, end, reverse=False):
        with self.context():
            self._seed(prompt)
            for result in self.predictor.propagate_in_video(
                self.state, start_frame_idx=prompt["frame_index"],
                max_frame_num_to_track=abs(end - prompt["frame_index"]), reverse=reverse,
            ):
                index = int(result[0])
                yield self._mask(result, prompt if index == prompt["frame_index"] else None)


def propagate_masks(segmenter, prompts, frame_count, folder):
    """Track each anchor to its neighbors in both directions; keep nearest anchor.

    Each direction starts with fresh memory. Overlaps measure disagreement and
    prompt corrections can be reproduced without previous browser preview state.
    """
    folder.mkdir()
    rows = [dict(frame_index=i, anchor=None, distance=frame_count, min_overlap_iou=1.0)
            for i in range(frame_count)]
    for pos, prompt in enumerate(prompts):
        anchor = prompt["frame_index"]
        left = prompts[pos - 1]["frame_index"] if pos else 0
        right = prompts[pos + 1]["frame_index"] if pos + 1 < len(prompts) else frame_count - 1
        print(f"Tracking anchor {anchor}: frames {left}..{right}", flush=True)
        for end, reverse in ((right, False), (left, True)):
            if reverse and anchor == left:
                continue
            for index, mask in segmenter.track(prompt, end, reverse):
                if not min(anchor, end) <= index <= max(anchor, end):
                    raise RuntimeError("SAM returned a frame outside the requested interval")
                row = rows[index]
                path = folder / f"{index:04d}.png"
                if path.exists():
                    with Image.open(path) as previous:
                        previous_mask = np.asarray(previous) > 0
                    union = np.count_nonzero(previous_mask | mask)
                    iou = np.count_nonzero(previous_mask & mask) / max(union, 1)
                    row["min_overlap_iou"] = min(row["min_overlap_iou"], float(iou))
                distance = abs(index - anchor)
                if row["anchor"] is None or distance < row["distance"]:
                    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                    row.update(anchor=anchor, distance=distance)
    if any(row["anchor"] is None for row in rows):
        raise RuntimeError("tracking failed to cover every frame")
    return rows


def prompt_browser(segmenter, frames_folder, document, port, *, model_label="SAM 2.1"):
    """Serve a local box/click editor and SAM previews; no source files are changed."""
    frame_count = len(document["images"])
    size = tuple(document["coordinate_size"])
    token = secrets.token_urlsafe(24)
    result = None
    done = False
    html = Path(__file__).with_name("rotgs_sam21_prompts.html").read_bytes().replace(
        b"SAM 2.1", model_label.encode("utf-8")
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data, content_type="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            # The crop editor and SAM editor share this local port. Allow opening
            # its landing page without copying the newly generated session URL.
            # Data requests and mutations still require the session token.
            if parsed.path == "/" and "token" not in parse_qs(parsed.query):
                self.send_response(302)
                self.send_header("Location", f"/?token={token}")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if parse_qs(parsed.query).get("token") != [token]:
                self.reply(b'{"error":"Session link expired or missing. Open / to enter the current editor."}', status=403)
                return
            if parsed.path == "/":
                self.reply(html, "text/html; charset=utf-8")
            elif parsed.path == "/config":
                self.reply(json.dumps(document).encode())
            elif parsed.path.startswith("/frame/"):
                try:
                    index = int(parsed.path.rsplit("/", 1)[1])
                    if not 0 <= index < frame_count:
                        raise ValueError()
                    self.reply((frames_folder / f"{index:04d}.jpg").read_bytes(), "image/jpeg")
                except ValueError:
                    self.reply(b'{}', status=404)
            else:
                self.reply(b'{}', status=404)

        def do_POST(self):
            nonlocal result, done
            if self.headers.get("X-SAM-Token") != token:
                self.reply(b'{}', status=403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/cancel":
                    done = True
                    self.reply(b'{}')
                    return
                frames = validate_prompts(payload, frame_count, size)
                if self.path == "/preview":
                    if len(frames) != 1:
                        raise ValueError("preview expects one anchor")
                    mask = segmenter.preview(frames[0])
                    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    overlay[mask] = [20, 220, 120, 110]
                    buffer = io.BytesIO()
                    Image.fromarray(overlay).save(buffer, format="PNG")
                    self.reply(buffer.getvalue(), "image/png")
                elif self.path == "/finish":
                    result = frames
                    done = True
                    self.reply(b'{}')
                else:
                    self.reply(b'{}', status=404)
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
                self.reply(json.dumps({"error": str(exc)}).encode(), status=400)

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        server.timeout = 1
        print(f"SAM prompt editor: http://localhost:{server.server_port}/?token={token}", flush=True)
        print(f"Forward port {server.server_port} in VS Code. Preview and correct the anchors, then continue.", flush=True)
        while not done:
            server.handle_request()
    if result is None:
        raise legacy.SelectionCancelled("SAM prompt editing cancelled")
    return result


def _write_review(staging, rows, ordered, *, model_label="SAM 2.1", prompts_filename="sam21_prompts.json"):
    import html

    review = staging / "mask_review"
    review.mkdir()
    cards = []
    for row, item in zip(rows, ordered):
        with Image.open(staging / "images" / item.output_name) as image:
            image.thumbnail((320, 320))
            background = Image.new("RGBA", image.size, (210, 210, 210, 255))
            background.alpha_composite(image)
            background.convert("RGB").save(review / item.output_name)
        warnings = []
        if row.get("detection_method") in ("text_and_box", "box"):
            warnings.append("box fallback: inspect silhouette")
        overlap = row["min_overlap_iou"]
        if overlap is not None and overlap < 0.90:
            warnings.append("anchor disagreement")
        if row["touches_crop_edge"]:
            warnings.append("touches crop edge")
        row["review_flags"] = warnings
        cards.append(
            f'<figure><a href="../images/{item.output_name}"><img src="{item.output_name}"></a>'
            f'<figcaption>{html.escape(item.output_name)} / {item.angle}° / '
            f'anchor {row["anchor"]}<br>Overlap IoU '
            f'{format(overlap, ".3f") if overlap is not None else "not evaluated (independent detection)"}'
            f'<br>{html.escape(", ".join(warnings))}</figcaption></figure>'
        )
    (review / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(model_label)} mask review</title>'
        '<style>body{font:16px system-ui;background:#20242b;color:white}'
        'main{display:flex;flex-wrap:wrap}figure{margin:12px}img{width:280px}</style>'
        '<h1>Product mask review</h1><p>Inspect all silhouettes and marker remnants. '
        'Overlap IoU compares predictions from adjacent anchors; it is not accuracy. '
        f'To correct a frame, add it to {html.escape(prompts_filename)} and rerun into a new output folder.</p><main>'
        + "".join(cards) + "</main>", encoding="utf-8",
    )


def _choose_crop(first, ordered, undistortion, segmentation_max_width, target_width, web_port, *, model_label="SAM 2.1"):
    """Supply the existing crop editor with six undistorted rotation views."""
    with tempfile.TemporaryDirectory(prefix="rotgs-sam21-crop-review-") as directory:
        comparisons = []
        for angle in tqdm(legacy.CROP_REVIEW_ANGLES, desc="Preparing crop comparison views"):
            item = min(ordered, key=lambda frame: abs(frame.angle - angle))
            path = Path(directory) / f"{angle:03d}.png"
            with Image.open(item.path) as source:
                with legacy._remap_rgb(source, undistortion) as corrected:
                    corrected.save(path)
            comparisons.append((angle, path))
        return legacy.choose_processing_settings_web(
            first, 12.0, segmentation_max_width, 1400, 900, True, False,
            target_width=target_width, port=web_port, comparison_images=comparisons,
            context_label=f"{model_label}: product crop (alpha will be replaced)",
        )


def prepare_rotgs_sequence_SAM21(
    folder_name, cameras_file, *, checkpoint, model="large", device="cuda",
    target_width=None, calibration_file=None, selection=None, destination_folder=None,
    prompts_json=None, box_only=False, anchor_interval=18, segmentation_max_width=1600,
    mask_threshold=0.0, web_port=8765, angle_token_from_right=None,
    aruco_side_length_mm=None,
):
    """Undistort/crop PNGs, prompt SAM, track, and publish a separate RGBA dataset.

    Selection uses full-resolution undistorted pixels. Saved SAM prompts use
    cropped working-resolution pixels, recorded explicitly in the JSON file.
    Source alpha is discarded, not used as an object-membership mask.
    """
    if model not in CONFIGS or device not in ("cuda", "cpu"):
        raise ValueError("invalid SAM model or device")
    if anchor_interval < 1 or segmentation_max_width < 1 or not np.isfinite(mask_threshold):
        raise ValueError("interval/working width must be positive; threshold must be finite")
    if target_width is not None and target_width <= 0:
        raise ValueError("target width must be positive")
    if not 1 <= web_port <= 65535:
        raise ValueError("web port must be between 1 and 65535")
    if box_only and prompts_json:
        raise ValueError("choose either --box-only or --prompts-json")
    if aruco_side_length_mm is not None and (
        not np.isfinite(aruco_side_length_mm) or aruco_side_length_mm <= 0
    ):
        raise ValueError("ArUco side length must be positive and finite")
    source = Path(folder_name).expanduser().resolve(strict=True)
    cameras = Path(cameras_file).expanduser().resolve(strict=True)
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("checkpoint must be a local SAM 2.1 checkpoint file")
    # Fail before an interactive selection if the optional dependency is missing.
    import importlib.util
    if importlib.util.find_spec("sam2") is None:
        raise RuntimeError("SAM 2.1 is not installed; see prepare_rotgs_sequence_SAM21.md")
    ordered, angle_index = legacy.detect_angle_order(
        sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
        angle_token_from_right=angle_token_from_right,
    )
    with Image.open(ordered[0].path) as image:
        source_size = image.size
    camera_text = cameras.read_text(encoding="utf-8")
    undistortion = legacy._resolve_undistortion(calibration_file, cameras, source_size, camera_text)
    with Image.open(ordered[0].path) as image:
        first = legacy._remap_rgb(image, undistortion)
    try:
        if selection is None:
            print("Select a crop containing the product at EVERY rotation angle, with a small margin.", flush=True)
            selection, crop, _, target_width = _choose_crop(
                first, ordered, undistortion, segmentation_max_width, target_width, web_port,
            )
        else:
            selection = tuple(selection)
            crop = legacy.crop_box_for_selection(selection, source_size, True)
    finally:
        first.close()
    legacy._validate_valid_crop(crop, undistortion)
    crop_size = (crop[2] - crop[0], crop[3] - crop[1])
    output_size = legacy.calculate_output_size(crop_size, target_width or crop_size[0])
    working_size = legacy.calculate_output_size(crop_size, min(segmentation_max_width, crop_size[0]))
    output = (Path(destination_folder).expanduser().resolve() if destination_folder else
              source.with_name(f"{source.name}_und_rn_roi_sqr_SAM21_{output_size[0]}"))
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
        version=1, coordinate_size=list(working_size),
        crop_box_left_top_right_bottom=list(crop), source_size=list(source_size),
        images=[item.path.name for item in ordered],
        frames=[dict(frame_index=i, box=box.copy(), points=[], labels=[]) for i in anchors],
    )
    if prompts_json:
        supplied = json.loads(Path(prompts_json).read_text(encoding="utf-8"))
        for key in ("images", "crop_box_left_top_right_bottom", "source_size"):
            if supplied.get(key) != document[key]:
                raise ValueError(f"saved prompts do not match this sequence/crop: {key}")
        document["frames"] = validate_prompts(supplied, len(ordered), working_size)

    with tempfile.TemporaryDirectory(prefix="rotgs-sam21-work-") as work_dir, \
         tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as staging_dir:
        work, staging = Path(work_dir), Path(staging_dir)
        frames_folder = work / "frames"
        frames_folder.mkdir()
        # Lossless native-resolution RGB is retained for final RGBA export.
        rgb_folder = work / "rgb"
        rgb_folder.mkdir()
        for index, item in enumerate(tqdm(ordered, desc="Undistorting source PNGs")):
            with Image.open(item.path) as image:
                if image.format != "PNG" or image.size != source_size:
                    raise ValueError(f"inconsistent source format/size: {item.path}")
                corrected = legacy._remap_rgb(image, undistortion)
            with corrected:
                with corrected.crop(crop) as cropped:
                    cropped.save(rgb_folder / item.output_name)
                    with cropped.resize(working_size, Image.Resampling.LANCZOS) as working:
                        # Official SAM video loader takes numbered JPEGs. Only these
                        # temporary inference copies use JPEG, never the exported RGB.
                        working.save(frames_folder / f"{index:04d}.jpg", quality=100, subsampling=0)
        if aruco_side_length_mm is not None:
            legacy._run_turntable_angle_estimation(
                source, undistortion.calibration_file, staging, aruco_side_length_mm,
            )
        segmenter = SAM21Segmenter(frames_folder, checkpoint, model, device, mask_threshold)
        if not prompts_json and not box_only:
            document["frames"] = prompt_browser(segmenter, frames_folder, document, web_port)
        prompts = validate_prompts(document, len(ordered), working_size)
        document["frames"] = prompts
        # Keep the interactive work even if subsequent tracking/export fails.
        recovery_file = output.with_name(f"{output.name}.prompts-{secrets.token_hex(4)}.json")
        with recovery_file.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
        print(f"Saved prompts (also retained on failure): {recovery_file}", flush=True)
        rows = propagate_masks(segmenter, prompts, len(ordered), work / "masks")
        images_folder = staging / "images"
        sparse_folder = staging / "sparse" / "0"
        images_folder.mkdir()
        sparse_folder.mkdir(parents=True)
        counts = dict(pixels=0, alpha_eq_0=0, alpha_eq_255=0, alpha_between_0_and_255=0)
        per_image = []
        for index, item in enumerate(tqdm(ordered, desc="Writing product RGBA PNGs")):
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
        _write_review(staging, rows, ordered)
        calibrated_text = legacy._cameras_text_with_matrix(
            camera_text, source_size, undistortion.output_camera_matrix,
        )
        transformed = legacy.transform_camera_intrinsics(calibrated_text, source_size, crop, output_size)
        (sparse_folder / "cameras.txt").write_text(transformed, encoding="utf-8")
        metadata = dict(
            input_folder=str(source), input_cameras_file=str(cameras),
            input_cameras_sha256=hashlib.sha256(camera_text.encode()).hexdigest(),
            source_format="PNG", source_size=list(source_size), image_count=len(ordered),
            alpha_mode="sam21_product", overwrite_alpha_mask=True,
            crop_box_left_top_right_bottom=list(crop), crop_size=list(crop_size),
            product_selection_left_top_right_bottom=list(selection), crop_mode="smallest_square",
            output_size=list(output_size), angles_degrees=[item.angle for item in ordered],
            angle_token_from_right=angle_index,
            renamed_images=[dict(source=item.path.name, angle_degrees=item.angle, output=item.output_name)
                            for item in ordered],
            undistortion=dict(applied=True, calibration_file=str(undistortion.calibration_file),
                              calibration_sha256=undistortion.calibration_sha256,
                              output_camera_matrix=undistortion.output_camera_matrix.tolist(),
                              alpha_remap="discarded_then_regenerated_after_undistortion"),
            segmentation=dict(backend="sam2.1", model=model, checkpoint=str(checkpoint),
                              working_size=list(working_size), mask_threshold=mask_threshold,
                              merge="nearest_anchor", independent_forward_reverse=True,
                              prompts_reviewed_in_browser=not prompts_json and not box_only,
                              source_alpha_used=False),
            alpha_statistics=legacy._with_percentages(counts), per_image_alpha_statistics=per_image,
            mask_review=rows,
        )
        (staging / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (staging / "sam21_prompts.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        # Publish only after all images, metadata, masks and optional angles succeed.
        if output.exists():
            raise FileExistsError(f"output appeared during processing: {output}")
        staging.rename(output)
    print(f"Prepared {len(ordered)} frames: {output}")
    print(f"Inspect masks: {output / 'mask_review' / 'index.html'}")
    return dict(output_folder=str(output), processed=len(ordered), metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder_name", help="original angle-named PNG sequence")
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path, help="local SAM 2.1 weights")
    parser.add_argument("--model", choices=CONFIGS, default="large")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--calibration", type=Path, dest="calibration_file")
    parser.add_argument("--width", type=int, dest="target_width")
    parser.add_argument("--selection", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="crop selection in full-resolution UNDISTORTED pixels; otherwise use browser")
    parser.add_argument("--output", type=Path, dest="destination_folder")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompts-json", type=Path, help="replay/edit saved sam21_prompts.json")
    prompt_group.add_argument("--box-only", action="store_true", help="skip prompt editor; seed all anchors with crop selection box")
    parser.add_argument("--anchor-interval", type=int, default=18, help="initial anchor spacing in frames (18 = 90 degrees)")
    parser.add_argument("--segmentation-max-width", type=int, default=1600)
    parser.add_argument("--mask-threshold", type=float, default=0.0, help="SAM logit threshold, not alpha probability")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--angle-token-from-right", type=int)
    parser.add_argument("--aruco-side-length-mm", type=float, help="optional angle estimation on original uncropped images")
    args = vars(parser.parse_args())
    args["cameras_file"] = args.pop("cameras")
    try:
        prepare_rotgs_sequence_SAM21(**args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyboardInterrupt) as exc:
        print(f"Error: {exc or 'cancelled'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
