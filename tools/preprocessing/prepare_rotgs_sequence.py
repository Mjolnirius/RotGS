"""Prepare one JPG turntable sequence for RotGS in a single pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


JPEG_EXTENSIONS = {".jpg", ".jpeg"}
SQUARE_OUTPUT_SUFFIX = "_rn_roi_sqr_PNGa"
SOURCE_ASPECT_OUTPUT_SUFFIX = "_rn_roi_PNGa_ds"
INTEGER_TOKEN_PATTERN = re.compile(r"(?<![\d.])-?\d+(?![\d.])")
CropBox = tuple[int, int, int, int]


class SelectionCancelled(RuntimeError):
    """Raised when an interactive selection or preview is cancelled."""


@dataclass(frozen=True)
class OrderedImage:
    """One source image with its detected angle and output filename."""

    path: Path
    angle: int
    output_name: str


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required; run this utility through the project "
            "environment with 'uv run python'"
        ) from exc
    return cv2


def _integer_tokens(path: Path) -> list[int]:
    """Return every standalone integer token from a filename stem."""
    return [int(match.group()) for match in INTEGER_TOKEN_PATTERN.finditer(path.stem)]


def detect_angle_order(
    image_files: list[Path],
    angle_token_from_right: int | None = None,
) -> tuple[list[OrderedImage], int]:
    """Detect the 5-degree filename token and return images in angle order.

    The returned token index is zero-based from the right. Automatic detection
    tests every numeric position that is present in every filename and accepts
    positions whose values form one complete 0, 5, 10, ... sequence.
    """
    if not image_files:
        raise ValueError("no JPG/JPEG files were found")

    token_lists = [_integer_tokens(path) for path in image_files]
    missing_tokens = [
        path.name for path, tokens in zip(image_files, token_lists) if not tokens
    ]
    if missing_tokens:
        raise ValueError(
            "filenames contain no integer tokens: " + ", ".join(missing_tokens[:5])
        )

    maximum_common_index = min(len(tokens) for tokens in token_lists) - 1
    if angle_token_from_right is not None:
        if not 0 <= angle_token_from_right <= maximum_common_index:
            raise ValueError(
                "angle_token_from_right must be between 0 and "
                f"{maximum_common_index}; got {angle_token_from_right}"
            )
        candidate_indices = [angle_token_from_right]
    else:
        candidate_indices = list(range(maximum_common_index + 1))

    valid_candidates: list[tuple[int, list[int]]] = []
    for index_from_right in candidate_indices:
        values = [tokens[-1 - index_from_right] for tokens in token_lists]
        sorted_values = sorted(values)
        if len(set(values)) != len(values):
            continue
        if sorted_values[0] != 0:
            continue
        if any(value < 0 or value > 360 or value % 5 for value in values):
            continue
        if sorted_values != list(range(0, sorted_values[-1] + 1, 5)):
            continue
        valid_candidates.append((index_from_right, values))

    if not valid_candidates:
        override_hint = (
            " Check --angle-token-from-right."
            if angle_token_from_right is not None
            else " Use --angle-token-from-right if automatic detection is ambiguous."
        )
        raise ValueError(
            "could not find a numeric filename component forming a complete "
            f"0, 5, 10, ... sequence.{override_hint}"
        )

    if len(valid_candidates) > 1:
        candidate_text = ", ".join(
            str(index) for index, _values in valid_candidates
        )
        raise ValueError(
            "multiple filename components look like 5-degree angle sequences "
            f"(indices from right: {candidate_text}); specify "
            "--angle-token-from-right"
        )

    detected_index, angles = valid_candidates[0]
    maximum_angle = max(angles)
    if maximum_angle not in {355, 360}:
        raise ValueError(
            "the detected sequence must cover a full rotation ending at 355 "
            f"or 360 degrees; detected maximum={maximum_angle}"
        )

    ordered_pairs = sorted(
        zip(image_files, angles),
        key=lambda item: (item[1], item[0].name.casefold(), item[0].name),
    )
    index_width = max(4, len(str(len(ordered_pairs) - 1)))
    ordered_images = [
        OrderedImage(path, angle, f"{index:0{index_width}d}.png")
        for index, (path, angle) in enumerate(ordered_pairs)
    ]
    return ordered_images, detected_index


def fit_selection_to_aspect(
    selection: CropBox,
    image_size: tuple[int, int],
) -> CropBox:
    """Expand a product selection around its center to the source aspect."""
    image_width, image_height = image_size
    left, top, right, bottom = selection
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image dimensions: {image_size}")
    if not (0 <= left < right <= image_width):
        raise ValueError(
            f"crop x-coordinates must satisfy 0 <= left < right <= {image_width}; "
            f"got left={left}, right={right}"
        )
    if not (0 <= top < bottom <= image_height):
        raise ValueError(
            f"crop y-coordinates must satisfy 0 <= top < bottom <= {image_height}; "
            f"got top={top}, bottom={bottom}"
        )

    target_aspect = image_width / image_height
    selection_width = right - left
    selection_height = bottom - top
    if selection_width / selection_height >= target_aspect:
        crop_width = selection_width
        crop_height = math.ceil(crop_width / target_aspect)
    else:
        crop_height = selection_height
        crop_width = math.ceil(crop_height * target_aspect)

    if crop_width > image_width or crop_height > image_height:
        return 0, 0, image_width, image_height

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_left = round(center_x - crop_width / 2.0)
    crop_top = round(center_y - crop_height / 2.0)
    crop_left = max(0, min(crop_left, image_width - crop_width))
    crop_top = max(0, min(crop_top, image_height - crop_height))
    return (
        crop_left,
        crop_top,
        crop_left + crop_width,
        crop_top + crop_height,
    )


def fit_selection_to_smallest_square(
    selection: CropBox,
    image_size: tuple[int, int],
) -> CropBox:
    """Fit the smallest in-image square containing the product selection.

    The crop is always made from real source pixels: it never stretches the
    image and never adds padding. Its side length is the larger selection
    dimension, and it is centered on the selection wherever image bounds allow.
    """
    image_width, image_height = image_size
    left, top, right, bottom = selection
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image dimensions: {image_size}")
    if not (0 <= left < right <= image_width):
        raise ValueError(
            f"crop x-coordinates must satisfy 0 <= left < right <= {image_width}; "
            f"got left={left}, right={right}"
        )
    if not (0 <= top < bottom <= image_height):
        raise ValueError(
            f"crop y-coordinates must satisfy 0 <= top < bottom <= {image_height}; "
            f"got top={top}, bottom={bottom}"
        )

    selection_width = right - left
    selection_height = bottom - top
    square_size = max(selection_width, selection_height)
    if square_size > image_width or square_size > image_height:
        raise ValueError(
            "the selected product box cannot fit inside an unpadded square crop: "
            f"selection={selection_width}x{selection_height}, "
            f"image={image_width}x{image_height}"
        )

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_left = round(center_x - square_size / 2.0)
    crop_top = round(center_y - square_size / 2.0)

    # Each crop origin must both remain inside the image and contain the full
    # selection. These bounds also handle a product selected near an edge.
    minimum_left = max(0, right - square_size)
    maximum_left = min(left, image_width - square_size)
    minimum_top = max(0, bottom - square_size)
    maximum_top = min(top, image_height - square_size)
    crop_left = max(minimum_left, min(crop_left, maximum_left))
    crop_top = max(minimum_top, min(crop_top, maximum_top))

    if not (
        crop_left <= left < right <= crop_left + square_size
        and crop_top <= top < bottom <= crop_top + square_size
    ):
        raise ValueError(
            "could not position the square crop without clipping the selected "
            f"product box {selection}"
        )
    return (
        crop_left,
        crop_top,
        crop_left + square_size,
        crop_top + square_size,
    )


def crop_box_for_selection(
    selection: CropBox,
    image_size: tuple[int, int],
    square_crop: bool,
) -> CropBox:
    """Create either the default tight square or legacy source-aspect crop."""
    if square_crop:
        return fit_selection_to_smallest_square(selection, image_size)
    return fit_selection_to_aspect(selection, image_size)


def calculate_output_size(
    crop_size: tuple[int, int],
    target_width: int,
) -> tuple[int, int]:
    """Return the exact target width and aspect-preserving integer height."""
    crop_width, crop_height = crop_size
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"invalid crop dimensions: {crop_size}")
    if target_width <= 0:
        raise ValueError(f"target width must be positive; got {target_width}")
    if target_width > crop_width:
        raise ValueError(
            f"target width {target_width} would upscale the {crop_width}-pixel crop"
        )
    return target_width, max(1, round(crop_height * target_width / crop_width))


def transform_camera_intrinsics(
    cameras_text: str,
    source_size: tuple[int, int],
    crop_box: CropBox,
    output_size: tuple[int, int],
) -> str:
    """Adjust COLMAP PINHOLE intrinsics for source scaling, crop, and resize."""
    source_width, source_height = source_size
    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top
    output_width, output_height = output_size
    updated_lines: list[str] = []
    camera_count = 0

    for line_number, line in enumerate(cameras_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            updated_lines.append(line)
            continue

        fields = stripped.split()
        if len(fields) < 5:
            raise ValueError(
                f"invalid cameras.txt entry on line {line_number}: {line!r}"
            )
        camera_id, model = fields[0], fields[1]
        if model != "PINHOLE":
            raise ValueError(
                f"unsupported camera model {model!r} on line {line_number}; "
                "this RotGS loader requires PINHOLE"
            )
        try:
            camera_width = int(fields[2])
            camera_height = int(fields[3])
            parameters = [float(value) for value in fields[4:]]
        except ValueError as exc:
            raise ValueError(
                f"invalid numeric camera value on line {line_number}: {line!r}"
            ) from exc
        if camera_width <= 0 or camera_height <= 0:
            raise ValueError(
                f"camera dimensions must be positive on line {line_number}"
            )
        if len(parameters) != 4:
            raise ValueError(
                f"PINHOLE requires exactly fx fy cx cy on line {line_number}"
            )
        if not math.isclose(
            camera_width / camera_height,
            source_width / source_height,
            rel_tol=0.01,
        ):
            raise ValueError(
                "camera and source image aspect ratios differ on line "
                f"{line_number}: camera={camera_width}x{camera_height}, "
                f"image={source_width}x{source_height}"
            )

        camera_to_source_x = source_width / camera_width
        camera_to_source_y = source_height / camera_height
        crop_to_output_x = output_width / crop_width
        crop_to_output_y = output_height / crop_height
        fx, fy, cx, cy = parameters
        updated_parameters = (
            fx * camera_to_source_x * crop_to_output_x,
            fy * camera_to_source_y * crop_to_output_y,
            (cx * camera_to_source_x - left) * crop_to_output_x,
            (cy * camera_to_source_y - top) * crop_to_output_y,
        )
        parameter_text = " ".join(
            format(value, ".17g") for value in updated_parameters
        )
        updated_lines.append(
            f"{camera_id} PINHOLE {output_width} {output_height} {parameter_text}"
        )
        camera_count += 1

    if camera_count == 0:
        raise ValueError("cameras.txt contains no camera entries")
    result = "\n".join(updated_lines)
    if cameras_text.endswith(("\n", "\r")):
        result += "\n"
    return result


def _display_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def select_product_box(
    image: Image.Image,
    display_max_width: int = 1400,
    display_max_height: int = 900,
) -> CropBox:
    """Interactively select the tight product box on the first image."""
    if display_max_width <= 0 or display_max_height <= 0:
        raise ValueError("display dimensions must be positive")
    cv2 = _import_cv2()
    display_rgb = _display_rgb(image)
    scale = min(
        1.0,
        display_max_width / image.width,
        display_max_height / image.height,
    )
    display_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
    if display_size != image.size:
        display_bgr = cv2.resize(
            display_bgr,
            display_size,
            interpolation=cv2.INTER_AREA,
        )

    window_name = "Select the product bounding box"
    print(
        "Draw a tight box around the product, then press Enter or Space. "
        "Press Esc to cancel."
    )
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, *display_size)
        x, y, width, height = (
            int(value)
            for value in cv2.selectROI(
                window_name,
                display_bgr,
                showCrosshair=True,
                fromCenter=False,
            )
        )
        cv2.destroyWindow(window_name)
    except cv2.error as exc:
        raise RuntimeError(
            "Could not open the ROI window. Run from a desktop session with GUI access."
        ) from exc
    if width <= 0 or height <= 0:
        raise SelectionCancelled("product selection cancelled")

    left = max(0, math.floor(x / scale))
    top = max(0, math.floor(y / scale))
    right = min(image.width, math.ceil((x + width) / scale))
    bottom = min(image.height, math.ceil((y + height) / scale))
    return left, top, right, bottom


def _scaled_box(box: CropBox, scale: float, size: tuple[int, int]) -> CropBox:
    width, height = size
    left, top, right, bottom = box
    return (
        max(0, min(width - 1, round(left * scale))),
        max(0, min(height - 1, round(top * scale))),
        max(1, min(width, round(right * scale))),
        max(1, min(height, round(bottom * scale))),
    )


def segment_product_alpha(
    image: Image.Image,
    product_box: CropBox,
    background_threshold: float = 12.0,
    working_max_width: int = 1600,
) -> Image.Image:
    """Create a foreground alpha mask for a product on a uniform background.

    The selected product box is used as a spatial prior. Background color is
    estimated from the crop border in CIE Lab space. The largest connected
    non-background component is retained, enclosed bright regions are filled,
    and resizing supplies antialiased edge alpha.
    """
    if not math.isfinite(background_threshold) or background_threshold <= 0:
        raise ValueError("background_threshold must be a positive finite number")
    if working_max_width <= 0:
        raise ValueError("working_max_width must be positive")
    left, top, right, bottom = product_box
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError(
            f"product box {product_box} lies outside image dimensions {image.size}"
        )

    cv2 = _import_cv2()
    rgb = _display_rgb(image)
    scale = min(1.0, working_max_width / image.width)
    work_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    if work_size != image.size:
        work_rgb = cv2.resize(rgb, work_size, interpolation=cv2.INTER_AREA)
    else:
        work_rgb = rgb
    work_height, work_width = work_rgb.shape[:2]
    work_box = _scaled_box(product_box, scale, work_size)

    lab = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_width = max(1, round(min(work_width, work_height) * 0.02))
    border_pixels = np.concatenate(
        (
            lab[:border_width, :, :].reshape(-1, 3),
            lab[-border_width:, :, :].reshape(-1, 3),
            lab[:, :border_width, :].reshape(-1, 3),
            lab[:, -border_width:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    background_lab = np.median(border_pixels, axis=0)
    color_distance = np.linalg.norm(lab - background_lab, axis=2)

    allowed = np.zeros((work_height, work_width), dtype=np.uint8)
    box_left, box_top, box_right, box_bottom = work_box
    margin = max(3, round(min(work_width, work_height) * 0.025))
    allowed[
        max(0, box_top - margin) : min(work_height, box_bottom + margin),
        max(0, box_left - margin) : min(work_width, box_right + margin),
    ] = 1
    foreground = ((color_distance > background_threshold) & (allowed > 0)).astype(
        np.uint8
    )

    kernel_size = max(3, round(min(work_width, work_height) * 0.004))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

    label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    if label_count <= 1:
        raise ValueError(
            "segmentation found no foreground; lower --background-threshold "
            "or select a tighter product box"
        )
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    foreground = (labels == largest_label).astype(np.uint8)

    padded = cv2.copyMakeBorder(
        foreground * 255,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    flooded = padded.copy()
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
    foreground = np.maximum(foreground * 255, holes)

    foreground_fraction = float(np.mean(foreground > 0))
    if not 0.002 <= foreground_fraction <= 0.95:
        raise ValueError(
            "implausible segmented foreground coverage: "
            f"{foreground_fraction:.2%}; adjust the ROI or background threshold"
        )

    alpha = Image.fromarray(foreground, mode="L")
    if alpha.size != image.size:
        resized_alpha = alpha.resize(image.size, Image.Resampling.BILINEAR)
        alpha.close()
        alpha = resized_alpha
    return alpha


def _segmentation_preview(
    image: Image.Image,
    alpha: Image.Image,
    threshold: float,
    display_max_width: int,
    display_max_height: int,
) -> int:
    """Show the first mask; return accept/reselect/cancel/threshold key."""
    cv2 = _import_cv2()
    rgb = _display_rgb(image)
    alpha_array = np.asarray(alpha, dtype=np.uint8)
    checker = np.full_like(rgb, 224)
    tile = max(8, min(image.size) // 40)
    yy, xx = np.indices(alpha_array.shape)
    checker[(xx // tile + yy // tile) % 2 == 0] = 176
    weight = alpha_array.astype(np.float32)[..., None] / 255.0
    composited = np.clip(rgb * weight + checker * (1.0 - weight), 0, 255).astype(
        np.uint8
    )
    mask_rgb = np.repeat(alpha_array[..., None], 3, axis=2)
    combined = np.concatenate((composited, mask_rgb), axis=1)
    scale = min(
        1.0,
        display_max_width / combined.shape[1],
        display_max_height / combined.shape[0],
    )
    display_size = (
        max(1, round(combined.shape[1] * scale)),
        max(1, round(combined.shape[0] * scale)),
    )
    display = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    if display_size != (combined.shape[1], combined.shape[0]):
        display = cv2.resize(display, display_size, interpolation=cv2.INTER_AREA)
    title = f"Segmentation threshold {threshold:g}"
    print(
        "Segmentation preview: Enter/Space=accept, R=reselect ROI, "
        "[=include more foreground, ]=include less, Esc/Q=cancel"
    )
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, *display_size)
        cv2.imshow(title, display)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(title)
    except cv2.error as exc:
        raise RuntimeError("Could not open the segmentation preview window") from exc
    return key


def choose_processing_settings(
    image: Image.Image,
    background_threshold: float,
    working_max_width: int,
    display_max_width: int,
    display_max_height: int,
    square_crop: bool,
) -> tuple[CropBox, CropBox, float]:
    """Select ROI and confirm segmentation on the first ordered image."""
    while True:
        selection = select_product_box(
            image,
            display_max_width=display_max_width,
            display_max_height=display_max_height,
        )
        crop_box = crop_box_for_selection(selection, image.size, square_crop)
        crop_left, crop_top, _crop_right, _crop_bottom = crop_box
        product_box = (
            selection[0] - crop_left,
            selection[1] - crop_top,
            selection[2] - crop_left,
            selection[3] - crop_top,
        )
        cropped = image.crop(crop_box)
        threshold = background_threshold
        try:
            while True:
                alpha = segment_product_alpha(
                    cropped,
                    product_box,
                    background_threshold=threshold,
                    working_max_width=working_max_width,
                )
                try:
                    key = _segmentation_preview(
                        cropped,
                        alpha,
                        threshold,
                        display_max_width,
                        display_max_height,
                    )
                finally:
                    alpha.close()
                if key in (13, 10, 32):
                    return selection, crop_box, threshold
                if key in (ord("r"), ord("R")):
                    break
                if key == ord("["):
                    threshold = max(1.0, threshold - 2.0)
                    continue
                if key == ord("]"):
                    threshold += 2.0
                    continue
                if key in (27, ord("q"), ord("Q")):
                    raise SelectionCancelled("segmentation preview cancelled")
        finally:
            cropped.close()


def _relative_product_box(selection: CropBox, crop_box: CropBox) -> CropBox:
    crop_left, crop_top, _right, _bottom = crop_box
    return (
        selection[0] - crop_left,
        selection[1] - crop_top,
        selection[2] - crop_left,
        selection[3] - crop_top,
    )


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nRotGS sequence preparation summary")
    print(f"  Input folder:             {summary['input_folder']}")
    print(f"  Input cameras:            {summary['input_cameras_file']}")
    print(f"  Output folder:            {summary['output_folder']}")
    print(f"  JPG images found:         {summary['found']}")
    print(f"  Successfully processed:   {summary['processed']}")
    print(f"  Detected angles:          {summary['angle_range']}")
    print(
        "  Angle token from right:  "
        f"{summary['angle_token_from_right']}"
    )
    print(f"  Source dimensions:        {summary['source_size']}")
    print(f"  Product box:              {summary['selection']}")
    print(f"  Crop mode:                {summary['crop_mode']}")
    print(f"  Applied crop:             {summary['crop_box']}")
    print(f"  Output dimensions:        {summary['output_size']}")
    print(
        "  Background threshold:   "
        f"{summary['background_threshold']:.2f} Lab units"
    )
    print(
        "  Foreground alpha:        "
        f"mean={summary['alpha_foreground_mean']:.2%}, "
        f"range={summary['alpha_foreground_min']:.2%}.."
        f"{summary['alpha_foreground_max']:.2%}"
    )
    print(f"  Output cameras:           {summary['output_cameras_file']}")
    print(f"  Metadata:                 {summary['metadata_file']}")
    print(f"  Elapsed time:             {summary['elapsed_seconds']:.3f} seconds")


def prepare_rotgs_sequence(
    folder_name: str | Path,
    cameras_file: str | Path,
    target_width: int = 540,
    *,
    selection: CropBox | None = None,
    background_threshold: float = 12.0,
    segmentation_max_width: int = 1600,
    display_max_width: int = 1400,
    display_max_height: int = 900,
    angle_token_from_right: int | None = None,
    confirm_segmentation: bool = True,
    square_crop: bool = True,
) -> dict[str, Any]:
    """Prepare JPGs and cameras in a sibling RotGS-ready output folder."""
    started_at = time.perf_counter()
    input_folder = Path(folder_name).expanduser().resolve(strict=True)
    input_cameras_file = Path(cameras_file).expanduser().resolve(strict=True)
    if not input_folder.is_dir():
        raise ValueError(f"input path is not a folder: {input_folder}")
    if not input_cameras_file.is_file():
        raise ValueError(f"cameras path is not a file: {input_cameras_file}")
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    if square_crop:
        copied_input_name = (
            input_folder.name
            if input_folder.name.casefold().endswith("_cpy")
            else f"{input_folder.name}_cpy"
        )
        output_name = f"{copied_input_name}{SQUARE_OUTPUT_SUFFIX}"
    else:
        output_name = f"{input_folder.name}{SOURCE_ASPECT_OUTPUT_SUFFIX}"
    output_folder = input_folder.with_name(output_name)
    if output_folder.exists():
        raise FileExistsError(
            f"output folder already exists; refusing to mix runs: {output_folder}"
        )

    image_files = sorted(
        (
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    ordered_images, detected_angle_index = detect_angle_order(
        image_files,
        angle_token_from_right=angle_token_from_right,
    )

    source_sizes: Counter[tuple[int, int]] = Counter()
    for ordered_image in ordered_images:
        with Image.open(ordered_image.path) as image:
            source_sizes[image.size] += 1
            if image.format not in {"JPEG", "MPO"}:
                raise ValueError(
                    f"input is not a JPEG image: {ordered_image.path.name}"
                )
    if len(source_sizes) != 1:
        size_text = ", ".join(
            f"{width}x{height} ({count})"
            for (width, height), count in sorted(source_sizes.items())
        )
        raise ValueError(f"all source images must have one resolution; found {size_text}")
    source_size = next(iter(source_sizes))

    cameras_text = input_cameras_file.read_text(encoding="utf-8")
    with Image.open(ordered_images[0].path) as first_source:
        first_source.load()
        first_rgb = first_source.convert("RGB")
    try:
        if selection is None:
            selection, crop_box, chosen_threshold = choose_processing_settings(
                first_rgb,
                background_threshold,
                segmentation_max_width,
                display_max_width,
                display_max_height,
                square_crop,
            )
        else:
            crop_box = crop_box_for_selection(selection, source_size, square_crop)
            chosen_threshold = background_threshold
            if confirm_segmentation:
                cropped_preview = first_rgb.crop(crop_box)
                product_box = _relative_product_box(selection, crop_box)
                try:
                    preview_alpha = segment_product_alpha(
                        cropped_preview,
                        product_box,
                        background_threshold=chosen_threshold,
                        working_max_width=segmentation_max_width,
                    )
                    try:
                        key = _segmentation_preview(
                            cropped_preview,
                            preview_alpha,
                            chosen_threshold,
                            display_max_width,
                            display_max_height,
                        )
                    finally:
                        preview_alpha.close()
                    if key not in (13, 10, 32):
                        raise SelectionCancelled("segmentation preview not accepted")
                finally:
                    cropped_preview.close()
    finally:
        first_rgb.close()

    crop_size = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    output_size = calculate_output_size(crop_size, target_width)
    product_box = _relative_product_box(selection, crop_box)
    updated_cameras_text = transform_camera_intrinsics(
        cameras_text,
        source_size,
        crop_box,
        output_size,
    )

    alpha_fractions: list[float] = []
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
            desc="Preparing RotGS sequence",
            unit="image",
        ):
            with Image.open(ordered_image.path) as source_image:
                source_image.load()
                if source_image.size != source_size:
                    raise RuntimeError(
                        f"source dimension changed for {ordered_image.path.name}"
                    )
                rgb_image = source_image.convert("RGB")
            try:
                cropped = rgb_image.crop(crop_box)
            finally:
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
            try:
                resized = rgba.resize(output_size, Image.Resampling.LANCZOS)
            finally:
                rgba.close()

            output_path = output_images_folder / ordered_image.output_name
            try:
                resized.save(output_path, format="PNG")
            finally:
                resized.close()
            with Image.open(output_path) as saved_image:
                saved_image.load()
                if saved_image.mode != "RGBA":
                    raise RuntimeError(
                        f"saved image is not RGBA: {ordered_image.output_name}"
                    )
                if saved_image.size != output_size:
                    raise RuntimeError(
                        f"saved image has wrong dimensions: {ordered_image.output_name}"
                    )
                alpha_array = np.asarray(saved_image.getchannel("A"), dtype=np.uint8)
                alpha_fractions.append(float(np.mean(alpha_array > 0)))

        metadata = {
            "input_folder": str(input_folder),
            "input_cameras_file": str(input_cameras_file),
            "input_cameras_sha256": hashlib.sha256(
                cameras_text.encode("utf-8")
            ).hexdigest(),
            "image_count": len(ordered_images),
            "angles_degrees": [item.angle for item in ordered_images],
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
            "background_threshold_lab": chosen_threshold,
            "segmentation_max_width": segmentation_max_width,
            "alpha_nonzero_fraction": {
                "mean": float(np.mean(alpha_fractions)),
                "minimum": min(alpha_fractions),
                "maximum": max(alpha_fractions),
            },
        }
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
        "angle_range": f"{ordered_images[0].angle}..{ordered_images[-1].angle}",
        "angle_token_from_right": detected_angle_index,
        "source_size": f"{source_size[0]}x{source_size[1]}",
        "selection": str(selection),
        "crop_mode": "smallest square" if square_crop else "source aspect",
        "crop_box": str(crop_box),
        "output_size": f"{output_size[0]}x{output_size[1]}",
        "background_threshold": chosen_threshold,
        "alpha_foreground_mean": float(np.mean(alpha_fractions)),
        "alpha_foreground_min": min(alpha_fractions),
        "alpha_foreground_max": max(alpha_fractions),
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    _print_summary(summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename a 5-degree JPG sequence, use the selected product box to "
            "position a square ROI, create foreground RGBA masks, downscale, "
            "and update cameras.txt."
        )
    )
    parser.add_argument("folder_name", help="folder containing source JPG files")
    parser.add_argument(
        "--cameras",
        required=True,
        help="session COLMAP cameras.txt file",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=540,
        help="final image width and, by default, height in pixels (default: 540)",
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
        "--background-threshold",
        type=float,
        default=12.0,
        help="initial Lab color distance from the border background (default: 12)",
    )
    parser.add_argument(
        "--segmentation-max-width",
        type=int,
        default=1600,
        help="maximum working width for mask detection (default: 1600)",
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
        "--angle-token-from-right",
        type=int,
        default=None,
        help="override the detected zero-based numeric token index from the right",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        prepare_rotgs_sequence(
            args.folder_name,
            args.cameras,
            target_width=args.width,
            background_threshold=args.background_threshold,
            segmentation_max_width=args.segmentation_max_width,
            display_max_width=args.display_max_width,
            display_max_height=args.display_max_height,
            angle_token_from_right=args.angle_token_from_right,
            square_crop=not args.keep_source_aspect,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
