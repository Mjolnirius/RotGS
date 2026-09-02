"""Prepare one PNG turntable sequence for RotGS in a single pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

if __package__:
    from .prepare_rotgs_sequence import (
        SOURCE_ASPECT_OUTPUT_SUFFIX,
        CropBox,
        SelectionCancelled,
        _relative_product_box,
        _segmentation_preview,
        calculate_output_size,
        choose_processing_settings,
        crop_box_for_selection,
        detect_angle_order,
        segment_product_alpha,
        select_product_box,
        transform_camera_intrinsics,
    )
else:
    from prepare_rotgs_sequence import (
        SOURCE_ASPECT_OUTPUT_SUFFIX,
        CropBox,
        SelectionCancelled,
        _relative_product_box,
        _segmentation_preview,
        calculate_output_size,
        choose_processing_settings,
        crop_box_for_selection,
        detect_angle_order,
        segment_product_alpha,
        select_product_box,
        transform_camera_intrinsics,
    )

if __package__:
    from .rotgs_web_selection import DEFAULT_WEB_PORT, choose_processing_settings_web
else:
    from rotgs_web_selection import DEFAULT_WEB_PORT, choose_processing_settings_web


PNG_EXTENSIONS = {".png"}
SOURCE_ALPHA_SQUARE_OUTPUT_SUFFIX = "_rn_roi_sqr_PNGaS"
REGENERATED_ALPHA_SQUARE_OUTPUT_SUFFIX = "_rn_roi_sqr_PNGaR"
CROP_REVIEW_ANGLES = (0, 60, 120, 180, 240, 300)


def _has_alpha(image: Image.Image) -> bool:
    """Return whether Pillow decoded an alpha band or PNG transparency data."""
    return "A" in image.getbands() or image.info.get("transparency") is not None


def _warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    print(f"Warning: {message}", file=sys.stderr)


def _untrusted_transparent_rgb_reason(image: Image.Image) -> str | None:
    """Detect common signs that RGB beneath source alpha is not trustworthy."""
    if not _has_alpha(image):
        return None

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    transparent = alpha == 0
    transparent_count = int(np.count_nonzero(transparent))
    if transparent_count >= max(100, round(alpha.size * 0.001)):
        transparent_rgb = rgb[transparent]
        near_zero_fraction = float(
            np.mean(np.max(transparent_rgb, axis=1) <= 1)
        )
        if near_zero_fraction >= 0.98:
            return (
                f"{near_zero_fraction:.1%} of fully transparent pixels have "
                "zero/near-zero RGB"
            )

    partial = (alpha > 0) & (alpha < 255)
    partial_count = int(np.count_nonzero(partial))
    if partial_count >= max(100, round(alpha.size * 0.001)):
        partial_rgb = rgb[partial].astype(np.int16)
        partial_alpha = alpha[partial].astype(np.int16)[:, None]
        bounded_fraction = float(
            np.mean(np.all(partial_rgb <= partial_alpha + 1, axis=1))
        )
        if bounded_fraction >= 0.995:
            return (
                f"{bounded_fraction:.1%} of partial-alpha pixels have RGB values "
                "bounded by alpha, suggesting premultiplied RGB"
            )
    return None


def _alpha_counts(alpha: np.ndarray) -> dict[str, int]:
    total = int(alpha.size)
    zero = int(np.count_nonzero(alpha == 0))
    opaque = int(np.count_nonzero(alpha == 255))
    return {
        "pixels": total,
        "alpha_eq_0": zero,
        "alpha_eq_255": opaque,
        "alpha_between_0_and_255": total - zero - opaque,
    }


def _with_percentages(counts: dict[str, int]) -> dict[str, Any]:
    total = counts["pixels"]
    result: dict[str, Any] = {"pixels": total}
    for key in ("alpha_eq_0", "alpha_between_0_and_255", "alpha_eq_255"):
        result[key] = {
            "count": counts[key],
            "percentage": 100.0 * counts[key] / total,
        }
    return result


def _alpha_warnings(counts: dict[str, int]) -> list[str]:
    total = counts["pixels"]
    zero = counts["alpha_eq_0"]
    partial = counts["alpha_between_0_and_255"]
    opaque = counts["alpha_eq_255"]
    warnings: list[str] = []
    nonopaque = zero + partial
    if opaque == total:
        warnings.append("every output pixel is fully opaque")
    elif nonopaque / total < 0.0001:
        warnings.append("fewer than 0.01% of output pixels contain transparency")
    if zero / total >= 0.99:
        warnings.append("at least 99% of output pixels are fully transparent")
    if partial / total >= 0.50:
        warnings.append("at least 50% of output pixels have partial alpha")
    return warnings


def _print_alpha_line(label: str, item: dict[str, Any]) -> None:
    print(
        f"  {label:<26}"
        f"{item['count']} ({item['percentage']:.4f}%)"
    )


def _print_summary(summary: dict[str, Any]) -> None:
    alpha = summary["alpha_statistics"]
    print("\nRotGS PNG sequence preparation summary")
    print(f"  Input folder:             {summary['input_folder']}")
    print(f"  Input cameras:            {summary['input_cameras_file']}")
    print(f"  Output folder:            {summary['output_folder']}")
    print(f"  PNG images found:         {summary['found']}")
    print(f"  Successfully processed:   {summary['processed']}")
    print(f"  Alpha mode:               {summary['alpha_mode']}")
    print(
        "  Source alpha files:       "
        f"{summary['source_alpha_files']}/{summary['found']}"
    )
    print(f"  Detected angles:          {summary['angle_range']}")
    print(f"  Angle token from right:   {summary['angle_token_from_right']}")
    print(f"  Source dimensions:        {summary['source_size']}")
    print(f"  Product box:              {summary['selection']}")
    print(f"  Crop mode:                {summary['crop_mode']}")
    print(f"  Applied crop:             {summary['crop_box']}")
    print(f"  Output dimensions:        {summary['output_size']}")
    if summary["background_threshold"] is not None:
        print(
            "  Background threshold:    "
            f"{summary['background_threshold']:.2f} Lab units"
        )
    _print_alpha_line("Alpha == 0:", alpha["alpha_eq_0"])
    _print_alpha_line("0 < alpha < 255:", alpha["alpha_between_0_and_255"])
    _print_alpha_line("Alpha == 255:", alpha["alpha_eq_255"])
    print(f"  Output cameras:           {summary['output_cameras_file']}")
    print(f"  Metadata:                 {summary['metadata_file']}")
    print(f"  Elapsed time:             {summary['elapsed_seconds']:.3f} seconds")


def prepare_rotgs_sequence_png(
    folder_name: str | Path,
    cameras_file: str | Path,
    target_width: int | None = None,
    *,
    selection: CropBox | None = None,
    background_threshold: float = 12.0,
    segmentation_max_width: int = 1600,
    display_max_width: int = 1400,
    display_max_height: int = 900,
    web_port: int = DEFAULT_WEB_PORT,
    angle_token_from_right: int | None = None,
    confirm_segmentation: bool = True,
    square_crop: bool = True,
    overwrite_alpha_mask: bool = False,
) -> dict[str, Any]:
    """Prepare PNGs and cameras in a sibling RotGS-ready output folder."""
    started_at = time.perf_counter()
    input_folder = Path(folder_name).expanduser().resolve(strict=True)
    input_cameras_file = Path(cameras_file).expanduser().resolve(strict=True)
    if not input_folder.is_dir():
        raise ValueError(f"input path is not a folder: {input_folder}")
    if not input_cameras_file.is_file():
        raise ValueError(f"cameras path is not a file: {input_cameras_file}")
    if target_width is not None and target_width <= 0:
        raise ValueError("target_width must be positive")

    if square_crop:
        output_suffix = (
            REGENERATED_ALPHA_SQUARE_OUTPUT_SUFFIX
            if overwrite_alpha_mask
            else SOURCE_ALPHA_SQUARE_OUTPUT_SUFFIX
        )
    else:
        output_suffix = SOURCE_ASPECT_OUTPUT_SUFFIX
    image_files = sorted(
        (
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in PNG_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not image_files:
        raise ValueError("no PNG files were found")
    ordered_images, detected_angle_index = detect_angle_order(
        image_files,
        angle_token_from_right=angle_token_from_right,
    )
    comparison_images = [
        (
            target_angle,
            min(ordered_images, key=lambda item: abs(item.angle - target_angle)).path,
        )
        for target_angle in CROP_REVIEW_ANGLES
    ]

    source_sizes: Counter[tuple[int, int]] = Counter()
    source_alpha_files = 0
    unreliable_rgb_files: list[tuple[str, str]] = []
    for ordered_image in tqdm(
        ordered_images,
        desc="Validating full-resolution source PNGs",
        unit="image",
    ):
        try:
            with Image.open(ordered_image.path) as image:
                image.load()
                if image.format != "PNG":
                    raise ValueError(
                        f"input is not PNG data: {ordered_image.path.name}"
                    )
                source_sizes[image.size] += 1
                has_alpha = _has_alpha(image)
                if has_alpha:
                    source_alpha_files += 1
                elif not overwrite_alpha_mask:
                    raise ValueError(
                        "input PNG has no alpha channel or transparency data: "
                        f"{ordered_image.path.name}"
                    )
                if overwrite_alpha_mask and has_alpha:
                    reason = _untrusted_transparent_rgb_reason(image)
                    if reason is not None:
                        unreliable_rgb_files.append((ordered_image.path.name, reason))
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"could not decode PNG image: {ordered_image.path.name}"
            ) from exc

    if len(source_sizes) != 1:
        size_text = ", ".join(
            f"{width}x{height} ({count})"
            for (width, height), count in sorted(source_sizes.items())
        )
        raise ValueError(
            f"all source images must have one resolution; found {size_text}"
        )
    source_size = next(iter(source_sizes))

    run_warnings: list[str] = []
    if unreliable_rgb_files:
        examples = "; ".join(
            f"{name}: {reason}" for name, reason in unreliable_rgb_files[:5]
        )
        extra = (
            f"; and {len(unreliable_rgb_files) - 5} more"
            if len(unreliable_rgb_files) > 5
            else ""
        )
        _warn(
            "regenerated segmentation may be unreliable because source RGB under "
            f"alpha appears destroyed or premultiplied ({examples}{extra}); missing "
            "RGB information will not be reconstructed",
            run_warnings,
        )

    cameras_text = input_cameras_file.read_text(encoding="utf-8")
    with Image.open(ordered_images[0].path) as first_source:
        first_source.load()
        first_image = first_source.convert(
            "RGB" if overwrite_alpha_mask else "RGBA"
        )
    try:
        needs_web_selection = selection is None or (
            overwrite_alpha_mask and confirm_segmentation
        )
        if needs_web_selection:
            (
                selection,
                crop_box,
                chosen_threshold,
                resolved_target_width,
            ) = choose_processing_settings_web(
                first_image,
                background_threshold,
                segmentation_max_width,
                display_max_width,
                display_max_height,
                square_crop,
                overwrite_alpha_mask,
                port=web_port,
                initial_selection=selection,
                target_width=target_width,
                comparison_images=comparison_images,
            )
        else:
            crop_box = crop_box_for_selection(selection, source_size, square_crop)
            chosen_threshold = background_threshold if overwrite_alpha_mask else None
            crop_width = crop_box[2] - crop_box[0]
            resolved_target_width = (
                crop_width if target_width is None else target_width
            )
    finally:
        first_image.close()

    crop_size = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    output_size = calculate_output_size(crop_size, resolved_target_width)
    output_name = f"{input_folder.name}{output_suffix}_{resolved_target_width}"
    output_folder = input_folder.with_name(output_name)
    if output_folder.exists():
        raise FileExistsError(
            "output folder already exists; refusing to mix runs: "
            f"{output_folder}"
        )
    product_box = _relative_product_box(selection, crop_box)
    updated_cameras_text = transform_camera_intrinsics(
        cameras_text,
        source_size,
        crop_box,
        output_size,
    )

    aggregate_counts = {
        "pixels": 0,
        "alpha_eq_0": 0,
        "alpha_eq_255": 0,
        "alpha_between_0_and_255": 0,
    }
    per_image_alpha: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        dir=input_folder.parent,
        prefix=f".{output_folder.name}.",
        suffix=".tmp",
    ) as temporary_directory:
        staging_folder = Path(temporary_directory)
        output_images_folder = staging_folder / "images"
        output_sparse_folder = staging_folder / "sparse" / "0"
        output_images_folder.mkdir(parents=True)
        output_sparse_folder.mkdir(parents=True)

        for ordered_image in tqdm(
            ordered_images,
            desc="Preparing RotGS PNG sequence",
            unit="image",
        ):
            with Image.open(ordered_image.path) as source_image:
                source_image.load()
                if source_image.format != "PNG":
                    raise RuntimeError(
                        f"source format changed for {ordered_image.path.name}"
                    )
                if source_image.size != source_size:
                    raise RuntimeError(
                        f"source dimension changed for {ordered_image.path.name}"
                    )
                if overwrite_alpha_mask:
                    rgb_image = source_image.convert("RGB")
                    cropped = rgb_image.crop(crop_box)
                    rgb_image.close()
                    try:
                        alpha = segment_product_alpha(
                            cropped,
                            product_box,
                            background_threshold=chosen_threshold,
                            working_max_width=segmentation_max_width,
                        )
                        try:
                            rgba = cropped.convert("RGBA")
                            rgba.putalpha(alpha)
                        finally:
                            alpha.close()
                    finally:
                        cropped.close()
                else:
                    if not _has_alpha(source_image):
                        raise RuntimeError(
                            f"source alpha disappeared for {ordered_image.path.name}"
                        )
                    source_rgba = source_image.convert("RGBA")
                    rgba = source_rgba.crop(crop_box)
                    source_rgba.close()

            try:
                if rgba.size == output_size:
                    resized = rgba.copy()
                else:
                    # Pillow resamples RGBA through premultiplied RGBa for Lanczos,
                    # keeping color and the unthresholded alpha channel aligned.
                    resized = rgba.resize(output_size, Image.Resampling.LANCZOS)
            finally:
                rgba.close()

            output_path = output_images_folder / ordered_image.output_name
            try:
                resized.save(output_path, format="PNG")
            finally:
                resized.close()

            try:
                with Image.open(output_path) as saved_image:
                    saved_image.load()
                    if saved_image.format != "PNG":
                        raise RuntimeError(
                            f"saved image is not PNG data: {ordered_image.output_name}"
                        )
                    if saved_image.mode != "RGBA" or "A" not in saved_image.getbands():
                        raise RuntimeError(
                            f"saved image is not RGBA: {ordered_image.output_name}"
                        )
                    if saved_image.size != output_size:
                        raise RuntimeError(
                            "saved image has wrong dimensions: "
                            f"{ordered_image.output_name}"
                        )
                    alpha_array = np.asarray(
                        saved_image.getchannel("A"),
                        dtype=np.uint8,
                    )
                    counts = _alpha_counts(alpha_array)
            except (UnidentifiedImageError, OSError) as exc:
                raise RuntimeError(
                    f"could not validate saved PNG: {ordered_image.output_name}"
                ) from exc

            for key in aggregate_counts:
                aggregate_counts[key] += counts[key]
            per_image_alpha.append(
                {
                    "image": ordered_image.output_name,
                    **_with_percentages(counts),
                }
            )

        alpha_statistics = _with_percentages(aggregate_counts)
        for warning in _alpha_warnings(aggregate_counts):
            _warn(warning, run_warnings)

        metadata: dict[str, Any] = {
            "input_folder": str(input_folder),
            "input_cameras_file": str(input_cameras_file),
            "input_cameras_sha256": hashlib.sha256(
                cameras_text.encode("utf-8")
            ).hexdigest(),
            "source_format": "PNG",
            "source_alpha_available": source_alpha_files == len(ordered_images),
            "source_alpha_files": source_alpha_files,
            "source_alpha_missing_files": len(ordered_images) - source_alpha_files,
            "alpha_mode": (
                "regenerated" if overwrite_alpha_mask else "preserve_source"
            ),
            "overwrite_alpha_mask": overwrite_alpha_mask,
            "image_count": len(ordered_images),
            "angles_degrees": [item.angle for item in ordered_images],
            "crop_review_images": [
                {
                    "angle_degrees": angle,
                    "source": path.name,
                }
                for angle, path in comparison_images
            ],
            "angle_token_from_right": detected_angle_index,
            "renamed_images": [
                {
                    "source": item.path.name,
                    "angle_degrees": item.angle,
                    "output": item.output_name,
                }
                for item in ordered_images
            ],
            "source_size": list(source_size),
            "product_selection_left_top_right_bottom": list(selection),
            "crop_mode": "smallest_square" if square_crop else "source_aspect",
            "crop_box_left_top_right_bottom": list(crop_box),
            "crop_size": list(crop_size),
            "output_size": list(output_size),
            "alpha_statistics": alpha_statistics,
            "per_image_alpha_statistics": per_image_alpha,
            "warnings": run_warnings,
        }
        if overwrite_alpha_mask:
            metadata["segmentation"] = {
                "background_threshold_lab": chosen_threshold,
                "segmentation_max_width": segmentation_max_width,
            }
        else:
            metadata["segmentation"] = None

        (output_sparse_folder / "cameras.txt").write_text(
            updated_cameras_text,
            encoding="utf-8",
        )
        (staging_folder / "preprocessing_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        staging_folder.rename(output_folder)

    summary: dict[str, Any] = {
        "input_folder": str(input_folder),
        "input_cameras_file": str(input_cameras_file),
        "output_folder": str(output_folder),
        "output_cameras_file": str(output_folder / "sparse" / "0" / "cameras.txt"),
        "metadata_file": str(output_folder / "preprocessing_metadata.json"),
        "found": len(ordered_images),
        "processed": len(ordered_images),
        "alpha_mode": "regenerated" if overwrite_alpha_mask else "preserve_source",
        "source_alpha_files": source_alpha_files,
        "angle_range": f"{ordered_images[0].angle}..{ordered_images[-1].angle}",
        "angle_token_from_right": detected_angle_index,
        "source_size": f"{source_size[0]}x{source_size[1]}",
        "selection": str(selection),
        "crop_mode": "smallest square" if square_crop else "source aspect",
        "crop_box": str(crop_box),
        "output_size": f"{output_size[0]}x{output_size[1]}",
        "background_threshold": chosen_threshold,
        "alpha_statistics": alpha_statistics,
        "warnings": run_warnings,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    _print_summary(summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename a 5-degree PNG sequence, crop it for RotGS, preserve its "
            "source alpha by default or regenerate alpha, optionally downscale, "
            "and update cameras.txt."
        )
    )
    parser.add_argument("folder_name", help="folder containing source PNG files")
    parser.add_argument(
        "--cameras",
        required=True,
        help="session COLMAP cameras.txt file",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "final image width in pixels; omit to preserve the selected crop's "
            "native resolution"
        ),
    )
    parser.add_argument(
        "--keep-source-aspect",
        action="store_true",
        help=(
            "use the previous source-aspect crop instead of the default smallest "
            "square crop"
        ),
    )
    parser.add_argument(
        "--overwrite-alpha-mask",
        action="store_true",
        help=(
            "ignore any source alpha and regenerate it with background segmentation"
        ),
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=12.0,
        help=(
            "initial Lab background distance when overwriting alpha (default: 12)"
        ),
    )
    parser.add_argument(
        "--segmentation-max-width",
        type=int,
        default=1600,
        help="maximum mask-detection width when overwriting alpha (default: 1600)",
    )
    parser.add_argument(
        "--display-max-width",
        type=int,
        default=1400,
        help="maximum interactive preview width (default: 1400)",
    )
    parser.add_argument(
        "--display-max-height",
        type=int,
        default=900,
        help="maximum interactive preview height (default: 900)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help=f"local browser-selection server port (default: {DEFAULT_WEB_PORT})",
    )
    parser.add_argument(
        "--angle-token-from-right",
        type=int,
        default=None,
        help="override the detected zero-based numeric token index from the right",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        prepare_rotgs_sequence_png(
            args.folder_name,
            args.cameras,
            target_width=args.width,
            background_threshold=args.background_threshold,
            segmentation_max_width=args.segmentation_max_width,
            display_max_width=args.display_max_width,
            display_max_height=args.display_max_height,
            angle_token_from_right=args.angle_token_from_right,
            square_crop=not args.keep_source_aspect,
            overwrite_alpha_mask=args.overwrite_alpha_mask,
            web_port=args.web_port,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
