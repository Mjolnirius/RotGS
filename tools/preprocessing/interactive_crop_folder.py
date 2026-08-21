"""Interactively select one aspect-preserving crop and apply it to PNGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".png"}
CropBox = tuple[int, int, int, int]


class SelectionCancelled(RuntimeError):
    """Raised when the interactive crop selection is cancelled."""


def fit_selection_to_aspect(
    selection: CropBox,
    image_size: tuple[int, int],
) -> CropBox:
    """Expand a selection around its center to the image's aspect ratio."""
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
    selected_width = right - left
    selected_height = bottom - top

    if selected_width / selected_height >= target_aspect:
        crop_width = selected_width
        crop_height = math.ceil(crop_width / target_aspect)
    else:
        crop_height = selected_height
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


def crop_image(image: Image.Image, crop_box: CropBox) -> Image.Image:
    """Return a cropped copy without resizing or changing color mode."""
    left, top, right, bottom = fit_selection_to_aspect(crop_box, image.size)
    return image.crop((left, top, right, bottom))


def update_camera_intrinsics(
    cameras_text: str,
    source_size: tuple[int, int],
    crop_box: CropBox,
) -> str:
    """Return COLMAP camera text adjusted for image scaling and cropping.

    RotGS accepts PINHOLE and SIMPLE_PINHOLE cameras. Intrinsics are first
    scaled from the camera file's dimensions to ``source_size`` and then the
    crop's left/top offsets are subtracted from the principal point.
    """
    source_width, source_height = source_size
    left, top, right, bottom = fit_selection_to_aspect(crop_box, source_size)
    output_width = right - left
    output_height = bottom - top
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
                f"invalid camera entry on line {line_number}: {line!r}"
            )

        camera_id, model = fields[0], fields[1]
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
        camera_aspect = camera_width / camera_height
        source_aspect = source_width / source_height
        if not math.isclose(camera_aspect, source_aspect, rel_tol=0.01):
            raise ValueError(
                "camera and PNG aspect ratios do not match on line "
                f"{line_number}: camera={camera_width}x{camera_height}, "
                f"PNG={source_width}x{source_height}"
            )

        scale_x = source_width / camera_width
        scale_y = source_height / camera_height
        if model == "PINHOLE":
            if len(parameters) != 4:
                raise ValueError(
                    f"PINHOLE requires fx fy cx cy on line {line_number}"
                )
            fx, fy, cx, cy = parameters
            updated_parameters = [
                fx * scale_x,
                fy * scale_y,
                cx * scale_x - left,
                cy * scale_y - top,
            ]
        elif model == "SIMPLE_PINHOLE":
            if len(parameters) != 3:
                raise ValueError(
                    f"SIMPLE_PINHOLE requires f cx cy on line {line_number}"
                )
            if not math.isclose(scale_x, scale_y, rel_tol=0.01):
                raise ValueError(
                    "SIMPLE_PINHOLE requires uniform image scaling on line "
                    f"{line_number}; scale_x={scale_x}, scale_y={scale_y}"
                )
            focal_scale = (scale_x + scale_y) / 2.0
            focal, cx, cy = parameters
            updated_parameters = [
                focal * focal_scale,
                cx * scale_x - left,
                cy * scale_y - top,
            ]
        else:
            raise ValueError(
                f"unsupported camera model {model!r} on line {line_number}; "
                "RotGS supports PINHOLE and SIMPLE_PINHOLE"
            )

        formatted_parameters = " ".join(
            format(value, ".17g") for value in updated_parameters
        )
        updated_lines.append(
            f"{camera_id} {model} {output_width} {output_height} "
            f"{formatted_parameters}"
        )
        camera_count += 1

    if camera_count == 0:
        raise ValueError("cameras.txt contains no camera entries")

    result = "\n".join(updated_lines)
    if cameras_text.endswith(("\n", "\r")):
        result += "\n"
    return result


def _display_rgb(image: Image.Image) -> np.ndarray:
    """Composite transparency on white and return an RGB display array."""
    if image.mode == "RGBA" or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, rgba).convert("RGB")
        rgba.close()
        return np.asarray(composited)
    return np.asarray(image.convert("RGB"))


def select_crop_box(
    image: Image.Image,
    display_max_width: int = 1400,
    display_max_height: int = 900,
) -> CropBox:
    """Interactively select and confirm an aspect-preserving crop box."""
    if display_max_width <= 0 or display_max_height <= 0:
        raise ValueError("display dimensions must be positive")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for interactive selection; run this utility "
            "through the project environment with 'uv run python'"
        ) from exc

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

    window_name = "Select product crop"
    while True:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, *display_size)
            roi = cv2.selectROI(
                window_name,
                display_bgr,
                showCrosshair=True,
                fromCenter=False,
            )
        except cv2.error as exc:
            raise RuntimeError(
                "Could not open the crop-selection window. Run the command "
                "from a desktop session with GUI access."
            ) from exc

        x, y, width, height = (int(value) for value in roi)
        if width <= 0 or height <= 0:
            cv2.destroyWindow(window_name)
            raise SelectionCancelled("crop selection cancelled")

        raw_box = (
            max(0, math.floor(x / scale)),
            max(0, math.floor(y / scale)),
            min(image.width, math.ceil((x + width) / scale)),
            min(image.height, math.ceil((y + height) / scale)),
        )
        crop_box = fit_selection_to_aspect(raw_box, image.size)

        preview = display_bgr.copy()
        left, top, right, bottom = crop_box
        display_box = (
            round(left * scale),
            round(top * scale),
            round(right * scale),
            round(bottom * scale),
        )
        overlay = preview.copy()
        overlay[: display_box[1], :] = 0
        overlay[display_box[3] :, :] = 0
        overlay[display_box[1] : display_box[3], : display_box[0]] = 0
        overlay[display_box[1] : display_box[3], display_box[2] :] = 0
        preview = cv2.addWeighted(preview, 0.35, overlay, 0.65, 0.0)
        cv2.rectangle(
            preview,
            (display_box[0], display_box[1]),
            (display_box[2] - 1, display_box[3] - 1),
            (0, 255, 0),
            2,
        )
        crop_width = right - left
        crop_height = bottom - top
        cv2.putText(
            preview,
            f"Final crop: {crop_width}x{crop_height}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "ENTER/SPACE: accept   R: reselect   ESC/Q: cancel",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, preview)

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (10, 13, 32):
                cv2.destroyWindow(window_name)
                return crop_box
            if key in (ord("r"), ord("R")):
                break
            if key in (27, ord("q"), ord("Q")):
                cv2.destroyWindow(window_name)
                raise SelectionCancelled("crop selection cancelled")
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise SelectionCancelled("crop-selection window was closed")


def _new_summary(
    input_folder: Path,
    output_folder: Path,
    input_cameras_file: Path,
) -> dict[str, Any]:
    return {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "input_cameras_file": str(input_cameras_file),
        "output_cameras_file": str(output_folder / "cameras.txt"),
        "found": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failures": [],
        "source_size": None,
        "crop_box": None,
        "output_size": None,
        "elapsed_seconds": 0.0,
        "folder_error": None,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nInteractive crop summary")
    print(f"  Input folder:             {summary['input_folder']}")
    print(f"  Output folder:            {summary['output_folder']}")
    print(f"  Input cameras file:       {summary['input_cameras_file']}")
    print(f"  Output cameras file:      {summary['output_cameras_file']}")
    print(f"  PNG images found:         {summary['found']}")
    print(f"  Successfully processed:   {summary['processed']}")
    print(f"  Skipped (verified):       {summary['skipped']}")
    print(f"  Failed:                   {summary['failed']}")
    print(f"  Source dimensions:        {summary['source_size'] or 'n/a'}")
    print(f"  Crop box (L,T,R,B):       {summary['crop_box'] or 'n/a'}")
    print(f"  Output dimensions:        {summary['output_size'] or 'n/a'}")
    print(f"  Elapsed time:             {summary['elapsed_seconds']:.3f} seconds")
    if summary["failures"]:
        print("  Failures:")
        for failure in summary["failures"]:
            print(f"    {failure}")
    if summary["folder_error"]:
        print(f"  Folder error:             {summary['folder_error']}")


def crop_png_folder(
    folder_name: str | Path,
    cameras_file: str | Path,
    crop_box: CropBox | None = None,
    display_max_width: int = 1400,
    display_max_height: int = 900,
) -> dict[str, Any]:
    """Select once and apply the same aspect-preserving crop to every PNG."""
    started_at = time.perf_counter()
    input_folder = Path(folder_name).expanduser()
    input_cameras_file = Path(cameras_file).expanduser()
    output_folder = input_folder.with_name(f"{input_folder.name}_zoomed")

    try:
        input_folder = input_folder.resolve(strict=True)
        input_cameras_file = input_cameras_file.resolve(strict=True)
        output_folder = input_folder.with_name(f"{input_folder.name}_zoomed")
    except (FileNotFoundError, OSError) as exc:
        summary = _new_summary(input_folder, output_folder, input_cameras_file)
        summary["folder_error"] = (
            f"Input folder does not exist or is inaccessible: {exc}"
        )
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    summary = _new_summary(input_folder, output_folder, input_cameras_file)
    if not input_folder.is_dir():
        summary["folder_error"] = "Input path exists but is not a folder."
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary
    if not input_cameras_file.is_file():
        summary["folder_error"] = "Input cameras path exists but is not a file."
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    try:
        cameras_text = input_cameras_file.read_text(encoding="utf-8")
    except OSError as exc:
        summary["folder_error"] = f"Could not read input cameras file: {exc}"
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    try:
        image_files = sorted(
            (
                path
                for path in input_folder.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    except OSError as exc:
        summary["folder_error"] = f"Could not inspect input folder: {exc}"
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    summary["found"] = len(image_files)
    if not image_files:
        summary["folder_error"] = "No PNG images were found in the input folder."
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    try:
        image_sizes: dict[Path, tuple[int, int]] = {}
        for image_path in image_files:
            with Image.open(image_path) as image:
                image_sizes[image_path] = image.size
    except Exception as exc:
        summary["folder_error"] = f"Could not inspect PNG dimensions: {exc}"
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    unique_sizes = set(image_sizes.values())
    if len(unique_sizes) != 1:
        formatted_sizes = ", ".join(
            f"{width}x{height}" for width, height in sorted(unique_sizes)
        )
        summary["folder_error"] = (
            "All images must have identical dimensions before applying one "
            f"pixel crop; found: {formatted_sizes}"
        )
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    source_size = next(iter(unique_sizes))
    summary["source_size"] = f"{source_size[0]}x{source_size[1]}"
    try:
        if crop_box is None:
            with Image.open(image_files[0]) as first_image:
                first_image.load()
                crop_box = select_crop_box(
                    first_image,
                    display_max_width,
                    display_max_height,
                )
        crop_box = fit_selection_to_aspect(crop_box, source_size)
    except (ValueError, RuntimeError, SelectionCancelled) as exc:
        summary["folder_error"] = str(exc)
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    left, top, right, bottom = crop_box
    output_size = right - left, bottom - top
    summary["crop_box"] = str(crop_box)
    summary["output_size"] = f"{output_size[0]}x{output_size[1]}"

    try:
        updated_cameras_text = update_camera_intrinsics(
            cameras_text,
            source_size,
            crop_box,
        )
    except ValueError as exc:
        summary["folder_error"] = str(exc)
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    try:
        output_folder.mkdir(exist_ok=True)
    except OSError as exc:
        summary["folder_error"] = f"Could not create output folder: {exc}"
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    metadata = {
        "input_folder": str(input_folder),
        "input_cameras_file": str(input_cameras_file),
        "input_cameras_sha256": hashlib.sha256(cameras_text.encode("utf-8")).hexdigest(),
        "source_size": list(source_size),
        "crop_box_left_top_right_bottom": list(crop_box),
        "output_size": list(output_size),
        "aspect_ratio": source_size[0] / source_size[1],
    }
    metadata_path = output_folder / "crop_metadata.json"
    existing_outputs = [
        output_folder / source_path.name
        for source_path in image_files
        if (output_folder / source_path.name).exists()
    ]
    if metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            summary["folder_error"] = f"Could not read existing crop metadata: {exc}"
            summary["elapsed_seconds"] = time.perf_counter() - started_at
            print(f"Error: {summary['folder_error']}", file=sys.stderr)
            _print_summary(summary)
            return summary
        if existing_metadata != metadata:
            summary["folder_error"] = (
                "The output folder was created with a different crop. Choose "
                "another input/output name or remove the old derived folder first."
            )
            summary["elapsed_seconds"] = time.perf_counter() - started_at
            print(f"Error: {summary['folder_error']}", file=sys.stderr)
            _print_summary(summary)
            return summary
    elif existing_outputs:
        summary["folder_error"] = (
            "The output folder contains PNGs but no crop_metadata.json, so their "
            "crop cannot be verified."
        )
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    output_cameras_file = output_folder / "cameras.txt"
    if output_cameras_file.exists():
        try:
            existing_cameras_text = output_cameras_file.read_text(encoding="utf-8")
        except OSError as exc:
            summary["folder_error"] = f"Could not read existing cameras.txt: {exc}"
            summary["elapsed_seconds"] = time.perf_counter() - started_at
            print(f"Error: {summary['folder_error']}", file=sys.stderr)
            _print_summary(summary)
            return summary
        if existing_cameras_text != updated_cameras_text:
            summary["folder_error"] = (
                "Existing output cameras.txt does not match the selected crop."
            )
            summary["elapsed_seconds"] = time.perf_counter() - started_at
            print(f"Error: {summary['folder_error']}", file=sys.stderr)
            _print_summary(summary)
            return summary
    else:
        temporary_cameras_file = output_folder / (
            f".cameras.{uuid.uuid4().hex}.tmp.txt"
        )
        try:
            temporary_cameras_file.write_text(
                updated_cameras_text,
                encoding="utf-8",
            )
            temporary_cameras_file.rename(output_cameras_file)
        except OSError as exc:
            if temporary_cameras_file.exists():
                temporary_cameras_file.unlink(missing_ok=True)
            summary["folder_error"] = f"Could not write output cameras.txt: {exc}"
            summary["elapsed_seconds"] = time.perf_counter() - started_at
            print(f"Error: {summary['folder_error']}", file=sys.stderr)
            _print_summary(summary)
            return summary

    for source_path in tqdm(image_files, desc="Cropping PNG images", unit="image"):
        output_path = output_folder / source_path.name
        temporary_path: Path | None = None
        try:
            with Image.open(source_path) as source_image:
                source_image.load()
                source_mode = source_image.mode

                if output_path.exists():
                    with Image.open(output_path) as existing_image:
                        existing_image.load()
                        if existing_image.size != output_size:
                            raise RuntimeError(
                                "existing output dimension mismatch: "
                                f"expected={output_size}, output={existing_image.size}"
                            )
                        if existing_image.mode != source_mode:
                            raise RuntimeError(
                                "existing output mode mismatch: "
                                f"expected={source_mode!r}, output={existing_image.mode!r}"
                            )
                    summary["skipped"] += 1
                    tqdm.write(f"Skipped verified existing output: {output_path}")
                    continue

                cropped_image = source_image.crop(crop_box)
                temporary_path = output_folder / (
                    f".{source_path.stem}.{uuid.uuid4().hex}.tmp.png"
                )
                try:
                    cropped_image.save(temporary_path, format="PNG")
                finally:
                    cropped_image.close()

            with Image.open(temporary_path) as saved_image:
                saved_image.load()
                if saved_image.size != output_size:
                    raise RuntimeError(
                        "saved output dimension mismatch: "
                        f"expected={output_size}, output={saved_image.size}"
                    )
                if saved_image.mode != source_mode:
                    raise RuntimeError(
                        "saved output mode mismatch: "
                        f"expected={source_mode!r}, output={saved_image.mode!r}"
                    )

            temporary_path.rename(output_path)
            temporary_path = None
            summary["processed"] += 1
        except Exception as exc:  # Continue processing independent images.
            summary["failed"] += 1
            failure = f"{source_path.name}: {exc}"
            summary["failures"].append(failure)
            tqdm.write(f"Failed: {failure}")
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError as cleanup_error:
                    tqdm.write(
                        f"Warning: could not remove temporary file {temporary_path}: "
                        f"{cleanup_error}"
                    )

    if not metadata_path.exists():
        try:
            metadata_path.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            summary["failed"] += 1
            failure = f"crop_metadata.json: {exc}"
            summary["failures"].append(failure)
            print(f"Failed: {failure}", file=sys.stderr)

    summary["elapsed_seconds"] = time.perf_counter() - started_at
    _print_summary(summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively select a product crop on the first PNG and apply "
            "it to every PNG while retaining the source aspect ratio."
        )
    )
    parser.add_argument("folder_name", help="Folder containing PNG images")
    parser.add_argument(
        "--cameras",
        required=True,
        help="input COLMAP cameras.txt file to adjust for scaling and cropping",
    )
    parser.add_argument(
        "--display-max-width",
        type=int,
        default=1400,
        help="maximum selection-window image width (default: 1400)",
    )
    parser.add_argument(
        "--display-max-height",
        type=int,
        default=900,
        help="maximum selection-window image height (default: 900)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = crop_png_folder(
        args.folder_name,
        args.cameras,
        display_max_width=args.display_max_width,
        display_max_height=args.display_max_height,
    )
    return 1 if summary["folder_error"] or summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
