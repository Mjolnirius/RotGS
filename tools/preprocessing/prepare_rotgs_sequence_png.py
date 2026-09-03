"""Prepare one PNG turntable sequence for RotGS in a single pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
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
    from .rotgs_undistortion_review import review_undistortion_web
    from .rotgs_web_selection import DEFAULT_WEB_PORT, choose_processing_settings_web
else:
    from rotgs_undistortion_review import review_undistortion_web
    from rotgs_web_selection import DEFAULT_WEB_PORT, choose_processing_settings_web


PNG_EXTENSIONS = {".png"}
SOURCE_ALPHA_SQUARE_OUTPUT_SUFFIX = "_rn_roi_sqr_PNGaS"
REGENERATED_ALPHA_SQUARE_OUTPUT_SUFFIX = "_rn_roi_sqr_PNGaR"
CROP_REVIEW_ANGLES = (0, 60, 120, 180, 240, 300)
UNDISTORTED_OUTPUT_MARKER = "_und"


@dataclass(frozen=True)
class Undistortion:
    """Validated full-resolution pinhole undistortion state."""

    calibration_file: Path
    calibration_sha256: str
    source_camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    output_camera_matrix: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    valid_pixels: np.ndarray


def _pinhole_camera_entries(cameras_text: str) -> list[tuple[int, int, np.ndarray]]:
    """Read image dimensions and K from every COLMAP PINHOLE entry."""
    entries: list[tuple[int, int, np.ndarray]] = []
    for line_number, line in enumerate(cameras_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 8 or fields[1] != "PINHOLE":
            raise ValueError(
                "undistortion requires PINHOLE cameras.txt entries with exactly "
                f"fx fy cx cy; invalid line {line_number}: {line!r}"
            )
        try:
            width, height = int(fields[2]), int(fields[3])
            fx, fy, cx, cy = (float(value) for value in fields[4:])
        except ValueError as exc:
            raise ValueError(
                f"invalid numeric camera value on line {line_number}: {line!r}"
            ) from exc
        entries.append(
            (
                width,
                height,
                np.array(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
            )
        )
    if not entries:
        raise ValueError("cameras.txt contains no PINHOLE camera entries")
    return entries


def _cameras_text_with_matrix(
    cameras_text: str,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
) -> str:
    """Replace PINHOLE dimensions and intrinsics while retaining comments/IDs."""
    width, height = image_size
    values = (
        float(camera_matrix[0, 0]),
        float(camera_matrix[1, 1]),
        float(camera_matrix[0, 2]),
        float(camera_matrix[1, 2]),
    )
    parameters = " ".join(format(value, ".17g") for value in values)
    updated: list[str] = []
    for line in cameras_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            updated.append(line)
            continue
        fields = stripped.split()
        updated.append(f"{fields[0]} PINHOLE {width} {height} {parameters}")
    result = "\n".join(updated)
    if cameras_text.endswith(("\n", "\r")):
        result += "\n"
    return result


def _load_undistortion(
    calibration_file: str | Path,
    source_size: tuple[int, int],
    cameras_text: str,
) -> Undistortion:
    """Validate an OpenCV calibration and build reusable full-size remap maps."""
    calibration_path = Path(calibration_file).expanduser().resolve(strict=True)
    if not calibration_path.is_file():
        raise ValueError(f"calibration path is not a file: {calibration_path}")
    try:
        with np.load(calibration_path, allow_pickle=False) as calibration:
            required = {"camera_matrix", "dist_coeffs", "image_size"}
            missing = required.difference(calibration.files)
            if missing:
                raise ValueError(
                    "calibration file is missing: " + ", ".join(sorted(missing))
                )
            camera_matrix = np.asarray(
                calibration["camera_matrix"], dtype=np.float64
            )
            distortion = np.asarray(
                calibration["dist_coeffs"], dtype=np.float64
            ).reshape(-1)
            calibration_size_array = np.asarray(
                calibration["image_size"]
            ).reshape(-1)
    except OSError as exc:
        raise ValueError(f"could not read calibration file: {calibration_path}") from exc

    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise ValueError("calibration camera_matrix must be a finite 3x3 matrix")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError("calibration focal lengths must be positive")
    if not np.allclose(camera_matrix[2], (0.0, 0.0, 1.0), atol=1e-12):
        raise ValueError("calibration camera_matrix must have final row [0, 0, 1]")
    if distortion.size not in {4, 5, 8, 12, 14} or not np.all(
        np.isfinite(distortion)
    ):
        raise ValueError(
            "calibration dist_coeffs must contain 4, 5, 8, 12, or 14 finite values"
        )
    if calibration_size_array.size != 2:
        raise ValueError("calibration image_size must contain width and height")
    calibration_size = tuple(int(value) for value in calibration_size_array)
    if calibration_size != source_size:
        raise ValueError(
            "calibration and source image dimensions differ: "
            f"calibration={calibration_size[0]}x{calibration_size[1]}, "
            f"source={source_size[0]}x{source_size[1]}"
        )

    for camera_width, camera_height, cameras_matrix in _pinhole_camera_entries(
        cameras_text
    ):
        if (camera_width, camera_height) != source_size:
            raise ValueError(
                "cameras.txt and source image dimensions differ: "
                f"camera={camera_width}x{camera_height}, "
                f"source={source_size[0]}x{source_size[1]}"
            )
        if not np.allclose(cameras_matrix, camera_matrix, rtol=1e-7, atol=1e-4):
            raise ValueError(
                "cameras.txt intrinsics do not match calibration camera_matrix"
            )

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for undistortion; run through 'uv run python'"
        ) from exc
    # Reusing K preserves central pixel density. Invalid outer pixels are tracked
    # and forbidden inside the accepted crop.
    output_camera_matrix = camera_matrix.copy()
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        output_camera_matrix,
        source_size,
        cv2.CV_32FC1,
    )
    source_width, source_height = source_size
    valid_pixels = (
        (map_x >= 0.0)
        & (map_x <= source_width - 1)
        & (map_y >= 0.0)
        & (map_y <= source_height - 1)
    )
    return Undistortion(
        calibration_file=calibration_path,
        calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        source_camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        output_camera_matrix=output_camera_matrix,
        map_x=map_x,
        map_y=map_y,
        valid_pixels=valid_pixels,
    )


def _resolve_undistortion(
    calibration_file: str | Path | None,
    cameras_file: Path,
    source_size: tuple[int, int],
    cameras_text: str,
) -> Undistortion:
    """Load an override or auto-discover the compatible session calibration."""
    if calibration_file is not None:
        print(
            f"Using explicit camera calibration: {calibration_file}",
            flush=True,
        )
        return _load_undistortion(calibration_file, source_size, cameras_text)

    session_root = next(
        (
            parent
            for parent in cameras_file.parents
            if parent.name.casefold().startswith("session_")
        ),
        cameras_file.parent,
    )
    candidates = sorted(
        path.resolve()
        for path in session_root.rglob("camera_calibration.npz")
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "undistortion is enabled by default, but no camera_calibration.npz "
            f"was found below {session_root}; pass --calibration only when the "
            "calibration is stored elsewhere"
        )

    compatible: list[Undistortion] = []
    rejected: list[tuple[Path, str]] = []
    for candidate in candidates:
        try:
            compatible.append(
                _load_undistortion(candidate, source_size, cameras_text)
            )
        except (OSError, ValueError) as exc:
            rejected.append((candidate, str(exc)))

    if len(compatible) == 1:
        selected = compatible[0]
        print(
            f"Automatically selected camera calibration: "
            f"{selected.calibration_file}",
            flush=True,
        )
        return selected
    if len(compatible) > 1:
        choices = "\n  ".join(
            str(item.calibration_file) for item in compatible
        )
        raise ValueError(
            "multiple camera calibrations match the source images and "
            f"cameras.txt; select one with --calibration:\n  {choices}"
        )

    reasons = "\n  ".join(
        f"{candidate}: {reason}" for candidate, reason in rejected
    )
    raise ValueError(
        "undistortion is enabled by default, but no discovered calibration "
        f"matches the source images and cameras.txt:\n  {reasons}"
    )


def _remap_rgb(image: Image.Image, undistortion: Undistortion) -> Image.Image:
    """Undistort RGB, filling outside-calibration pixels with white."""
    import cv2

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    corrected = cv2.remap(
        rgb,
        undistortion.map_x,
        undistortion.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(corrected)


def _remap_rgba(image: Image.Image, undistortion: Undistortion) -> Image.Image:
    """Undistort RGBA in premultiplied form to prevent transparent-edge halos."""
    import cv2

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    alpha_u16 = alpha.astype(np.uint16)
    premultiplied = np.empty_like(rgb)
    for channel in range(3):
        product = rgb[..., channel].astype(np.uint16) * alpha_u16
        premultiplied[..., channel] = ((product + 127) // 255).astype(np.uint8)
    corrected_premultiplied = cv2.remap(
        premultiplied,
        undistortion.map_x,
        undistortion.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    corrected_alpha = cv2.remap(
        alpha,
        undistortion.map_x,
        undistortion.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    corrected_rgb = np.zeros_like(corrected_premultiplied)
    nonzero = corrected_alpha > 0
    denominator = corrected_alpha[nonzero].astype(np.float32)
    for channel in range(3):
        values = corrected_premultiplied[..., channel][nonzero].astype(np.float32)
        corrected_rgb[..., channel][nonzero] = np.clip(
            np.rint(values * 255.0 / denominator), 0, 255
        ).astype(np.uint8)
    corrected = np.dstack((corrected_rgb, corrected_alpha))
    return Image.fromarray(corrected)


def _undistort_image(
    image: Image.Image,
    undistortion: Undistortion | None,
    overwrite_alpha_mask: bool,
) -> Image.Image:
    """Return an independent source image in the mode needed by its alpha branch."""
    if undistortion is None:
        return image.convert("RGB" if overwrite_alpha_mask else "RGBA")
    if overwrite_alpha_mask:
        return _remap_rgb(image, undistortion)
    return _remap_rgba(image, undistortion)


def _validate_valid_crop(
    crop_box: CropBox,
    undistortion: Undistortion | None,
) -> None:
    if undistortion is None:
        return
    left, top, right, bottom = crop_box
    valid_crop = undistortion.valid_pixels[top:bottom, left:right]
    invalid_count = int(valid_crop.size - np.count_nonzero(valid_crop))
    if invalid_count:
        invalid_percentage = 100.0 * invalid_count / valid_crop.size
        raise ValueError(
            "selected crop contains pixels outside the valid undistorted image "
            f"({invalid_count} pixels, {invalid_percentage:.4f}%); move or shrink "
            "the ROI"
        )


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
        "  Undistortion:             "
        + ("applied" if summary["undistortion_applied"] else "not requested")
    )
    if summary["calibration_file"] is not None:
        print(f"  Calibration:              {summary['calibration_file']}")
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
    calibration_file: str | Path | None = None,
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
    destination_folder: str | Path | None = None,
    review_label: str | None = None,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Prepare PNGs and cameras in a sibling RotGS-ready output folder."""
    started_at = time.perf_counter()
    input_folder = Path(folder_name).expanduser().resolve(strict=True)
    input_cameras_file = Path(cameras_file).expanduser().resolve(strict=True)
    if not input_folder.is_dir():
        raise ValueError(f"input path is not a folder: {input_folder}")
    if not input_cameras_file.is_file():
        raise ValueError(f"cameras path is not a file: {input_cameras_file}")
    requested_output_folder = (
        Path(destination_folder).expanduser().resolve()
        if destination_folder is not None
        else None
    )
    if requested_output_folder is not None:
        if requested_output_folder == input_folder:
            raise ValueError("output folder must differ from the input folder")
        if input_folder in requested_output_folder.parents:
            raise ValueError("output folder must not be inside the input folder")
        if not requested_output_folder.parent.is_dir():
            raise ValueError(
                f"output parent does not exist: {requested_output_folder.parent}"
            )
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
    cameras_text = input_cameras_file.read_text(encoding="utf-8")
    print("Preparing full-resolution undistortion maps...", flush=True)
    undistortion = _resolve_undistortion(
        calibration_file,
        input_cameras_file,
        source_size,
        cameras_text,
    )
    review_frames = [(item.angle, item.path) for item in ordered_images]
    accepted = review_undistortion_web(
        review_frames,
        lambda image: _undistort_image(
            image,
            undistortion,
            overwrite_alpha_mask,
        ),
        display_max_width,
        display_max_height,
        preserve_alpha=not overwrite_alpha_mask,
        destination_to_source_maps=(undistortion.map_x, undistortion.map_y),
        port=web_port,
        context_label=review_label,
    )
    if not accepted:
        raise SelectionCancelled(
            "undistortion was rejected; no output was written"
        )

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

    preview_temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    preview_comparison_images = comparison_images
    first_image: Image.Image | None = None
    try:
        if undistortion is not None:
            preview_temporary_directory = tempfile.TemporaryDirectory(
                prefix="rotgs-undistorted-previews-"
            )
            preview_root = Path(preview_temporary_directory.name)
            preview_comparison_images = []
            for index, (angle, source_path) in enumerate(
                tqdm(
                    comparison_images,
                    desc="Undistorting browser crop-review previews",
                    unit="view",
                )
            ):
                with Image.open(source_path) as comparison_source:
                    comparison_source.load()
                    corrected_comparison = _undistort_image(
                        comparison_source,
                        undistortion,
                        overwrite_alpha_mask,
                    )
                corrected_path = preview_root / f"{index:02d}.png"
                try:
                    corrected_comparison.save(corrected_path, format="PNG")
                finally:
                    corrected_comparison.close()
                preview_comparison_images.append((angle, corrected_path))

        with Image.open(ordered_images[0].path) as first_source:
            first_source.load()
            first_image = _undistort_image(
                first_source,
                undistortion,
                overwrite_alpha_mask,
            )
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
                comparison_images=preview_comparison_images,
                context_label=review_label,
            )
        else:
            crop_box = crop_box_for_selection(selection, source_size, square_crop)
            chosen_threshold = background_threshold if overwrite_alpha_mask else None
            crop_width = crop_box[2] - crop_box[0]
            resolved_target_width = (
                crop_width if target_width is None else target_width
            )
    finally:
        if first_image is not None:
            first_image.close()
        if preview_temporary_directory is not None:
            preview_temporary_directory.cleanup()

    _validate_valid_crop(crop_box, undistortion)
    crop_size = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    output_size = calculate_output_size(crop_size, resolved_target_width)
    undistortion_marker = UNDISTORTED_OUTPUT_MARKER if undistortion else ""
    output_name = (
        f"{input_folder.name}{undistortion_marker}"
        f"{output_suffix}_{resolved_target_width}"
    )
    output_folder = (
        requested_output_folder
        if requested_output_folder is not None
        else input_folder.with_name(output_name)
    )
    if output_folder.exists():
        raise FileExistsError(
            "output folder already exists; refusing to mix runs: "
            f"{output_folder}"
        )
    product_box = _relative_product_box(selection, crop_box)
    pinhole_source_text = (
        _cameras_text_with_matrix(
            cameras_text,
            source_size,
            undistortion.output_camera_matrix,
        )
        if undistortion is not None
        else cameras_text
    )
    updated_cameras_text = transform_camera_intrinsics(
        pinhole_source_text,
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
        dir=output_folder.parent,
        prefix=f".{output_folder.name}.",
        suffix=".tmp",
    ) as temporary_directory:
        staging_folder = Path(temporary_directory)
        output_images_folder = staging_folder / "images"
        output_sparse_folder = staging_folder / "sparse" / "0"
        output_images_folder.mkdir(parents=True)
        output_sparse_folder.mkdir(parents=True)

        progress_description = (
            "Undistorting and preparing RotGS PNG sequence"
            if undistortion is not None
            else "Preparing RotGS PNG sequence"
        )
        for ordered_image in tqdm(
            ordered_images,
            desc=progress_description,
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
                if not overwrite_alpha_mask and not _has_alpha(source_image):
                    raise RuntimeError(
                        f"source alpha disappeared for {ordered_image.path.name}"
                    )
                corrected_source = _undistort_image(
                    source_image,
                    undistortion,
                    overwrite_alpha_mask,
                )
                try:
                    if overwrite_alpha_mask:
                        cropped = corrected_source.crop(crop_box)
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
                        rgba = corrected_source.crop(crop_box)
                finally:
                    corrected_source.close()

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
            "review_label": review_label,
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
            "undistortion": None,
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
        if undistortion is not None:
            metadata["undistortion"] = {
                "applied": True,
                "calibration_file": str(undistortion.calibration_file),
                "calibration_sha256": undistortion.calibration_sha256,
                "source_camera_matrix": (
                    undistortion.source_camera_matrix.tolist()
                ),
                "distortion_coefficients": (
                    undistortion.distortion_coefficients.tolist()
                ),
                "output_camera_matrix": (
                    undistortion.output_camera_matrix.tolist()
                ),
                "output_policy": "same_intrinsics_preserve_central_density",
                "interpolation": "opencv_inter_linear",
                "valid_source_pixel_percentage": (
                    100.0 * float(np.mean(undistortion.valid_pixels))
                ),
                "accepted_crop_contains_only_valid_pixels": True,
                "alpha_remap": (
                    "discarded_then_regenerated_after_undistortion"
                    if overwrite_alpha_mask
                    else "premultiplied_rgba"
                ),
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
        "review_label": review_label,
        "input_cameras_file": str(input_cameras_file),
        "output_folder": str(output_folder),
        "output_cameras_file": str(output_folder / "sparse" / "0" / "cameras.txt"),
        "metadata_file": str(output_folder / "preprocessing_metadata.json"),
        "found": len(ordered_images),
        "processed": len(ordered_images),
        "alpha_mode": "regenerated" if overwrite_alpha_mask else "preserve_source",
        "undistortion_applied": undistortion is not None,
        "calibration_file": (
            str(undistortion.calibration_file) if undistortion is not None else None
        ),
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
    if print_summary:
        _print_summary(summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename a 5-degree PNG sequence, crop it for RotGS, preserve its "
            "source alpha by default or regenerate alpha, undistort by default "
            "and downscale, and update cameras.txt."
        )
    )
    parser.add_argument("folder_name", help="folder containing source PNG files")
    parser.add_argument(
        "--cameras",
        required=True,
        help="session COLMAP cameras.txt file",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "override the automatically discovered camera_calibration.npz; "
            "normally no argument is needed"
        ),
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
            calibration_file=args.calibration,
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
