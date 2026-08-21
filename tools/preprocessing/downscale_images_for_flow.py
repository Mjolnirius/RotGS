"""Create consistently sized RGBA PNGs for RotGS optical-flow training."""

from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".png"}


def calculate_downscaled_size(
    width: int,
    height: int,
    max_width: int = 1600,
) -> tuple[int, int]:
    """Return an aspect-preserving size whose width does not exceed max_width."""
    if width <= 0 or height <= 0:
        raise ValueError(
            f"image dimensions must be positive; got width={width}, height={height}"
        )
    if max_width <= 0:
        raise ValueError(f"max_width must be positive; got {max_width}")
    if width <= max_width:
        return width, height

    scale = max_width / width
    return max_width, max(1, round(height * scale))


def downscale_image_for_flow(
    image: Image.Image,
    max_width: int = 1600,
) -> Image.Image:
    """Return an RGBA copy resized with its alpha channel when necessary."""
    target_size = calculate_downscaled_size(image.width, image.height, max_width)
    rgba_image = image.copy() if image.mode == "RGBA" else image.convert("RGBA")

    if rgba_image.size == target_size:
        return rgba_image

    resized_image = rgba_image.resize(target_size, Image.Resampling.LANCZOS)
    rgba_image.close()
    return resized_image


def update_camera_intrinsics(
    cameras_text: str,
    output_size: tuple[int, int],
) -> str:
    """Return COLMAP camera intrinsics scaled to ``output_size``.

    RotGS reads PINHOLE and SIMPLE_PINHOLE camera models. Each camera entry is
    scaled independently from the width and height recorded in cameras.txt.
    """
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise ValueError(f"output dimensions must be positive; got {output_size}")

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
        if not math.isclose(
            camera_width / camera_height,
            output_width / output_height,
            rel_tol=0.01,
        ):
            raise ValueError(
                "camera and output image aspect ratios do not match on line "
                f"{line_number}: camera={camera_width}x{camera_height}, "
                f"output={output_width}x{output_height}"
            )

        scale_x = output_width / camera_width
        scale_y = output_height / camera_height
        if model == "PINHOLE":
            if len(parameters) != 4:
                raise ValueError(
                    f"PINHOLE requires fx fy cx cy on line {line_number}"
                )
            fx, fy, cx, cy = parameters
            updated_parameters = [
                fx * scale_x,
                fy * scale_y,
                cx * scale_x,
                cy * scale_y,
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
            focal, cx, cy = parameters
            focal_scale = (scale_x + scale_y) / 2.0
            updated_parameters = [
                focal * focal_scale,
                cx * scale_x,
                cy * scale_y,
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


def _new_summary(
    input_folder: Path,
    output_folder: Path,
    input_cameras_file: Path,
    max_width: int,
) -> dict[str, Any]:
    return {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "input_cameras_file": str(input_cameras_file),
        "output_cameras_file": str(output_folder / "cameras.txt"),
        "max_width": max_width,
        "found": 0,
        "processed": 0,
        "resized": 0,
        "already_within_limit": 0,
        "skipped": 0,
        "failed": 0,
        "failures": [],
        "source_resolution_counts": {},
        "output_resolution_counts": {},
        "elapsed_seconds": 0.0,
        "folder_error": None,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nOptical-flow image preparation summary")
    print(f"  Input folder:             {summary['input_folder']}")
    print(f"  Output folder:            {summary['output_folder']}")
    print(f"  Input cameras file:       {summary['input_cameras_file']}")
    print(f"  Output cameras file:      {summary['output_cameras_file']}")
    print(f"  Maximum width:            {summary['max_width']} px")
    print(f"  PNG images found:         {summary['found']}")
    print(f"  Successfully processed:   {summary['processed']}")
    print(f"  Downscaled:               {summary['resized']}")
    print(f"  Already within limit:     {summary['already_within_limit']}")
    print(f"  Skipped (verified):       {summary['skipped']}")
    print(f"  Failed:                   {summary['failed']}")
    print(f"  Elapsed time:             {summary['elapsed_seconds']:.3f} seconds")

    print("  Source resolutions:")
    if summary["source_resolution_counts"]:
        for resolution, count in summary["source_resolution_counts"].items():
            print(f"    {resolution}: {count}")
    else:
        print("    none")

    print("  Output resolutions:")
    if summary["output_resolution_counts"]:
        for resolution, count in summary["output_resolution_counts"].items():
            print(f"    {resolution}: {count}")
    else:
        print("    none")

    if summary["failures"]:
        print("  Failures:")
        for failure in summary["failures"]:
            print(f"    {failure}")
    if summary["folder_error"]:
        print(f"  Folder error:             {summary['folder_error']}")


def downscale_folder_for_flow(
    folder_name: str | Path,
    cameras_file: str | Path,
    max_width: int = 1600,
) -> dict[str, Any]:
    """Write flow-ready RGBA PNGs to ``<folder_name>_flow_ready``.

    PNG files directly inside ``folder_name`` are processed. Existing outputs
    are retained only if their dimensions and RGBA mode match the requested
    result. Source images are never modified.
    """
    if max_width <= 0:
        raise ValueError(f"max_width must be positive; got {max_width}")

    started_at = time.perf_counter()
    input_folder = Path(folder_name).expanduser()
    input_cameras_file = Path(cameras_file).expanduser()
    output_folder = input_folder.with_name(f"{input_folder.name}_flow_ready")

    try:
        input_folder = input_folder.resolve(strict=True)
        output_folder = input_folder.with_name(f"{input_folder.name}_flow_ready")
    except (FileNotFoundError, OSError) as exc:
        summary = _new_summary(
            input_folder,
            output_folder,
            input_cameras_file,
            max_width,
        )
        summary["folder_error"] = (
            f"Input folder does not exist or is inaccessible: {exc}"
        )
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    try:
        input_cameras_file = input_cameras_file.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        summary = _new_summary(
            input_folder,
            output_folder,
            input_cameras_file,
            max_width,
        )
        summary["folder_error"] = (
            f"Input cameras file does not exist or is inaccessible: {exc}"
        )
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    summary = _new_summary(
        input_folder,
        output_folder,
        input_cameras_file,
        max_width,
    )
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
    source_resolution_counts: Counter[tuple[int, int]] = Counter()
    output_resolution_counts: Counter[tuple[int, int]] = Counter()

    try:
        for source_path in image_files:
            with Image.open(source_path) as source_image:
                source_size = source_image.size
            target_size = calculate_downscaled_size(*source_size, max_width)
            source_resolution_counts[source_size] += 1
            output_resolution_counts[target_size] += 1
    except Exception as exc:
        summary["folder_error"] = f"Could not inspect image dimensions: {exc}"
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    summary["source_resolution_counts"] = {
        f"{width}x{height}": count
        for (width, height), count in sorted(source_resolution_counts.items())
    }
    summary["output_resolution_counts"] = {
        f"{width}x{height}": count
        for (width, height), count in sorted(output_resolution_counts.items())
    }

    if len(output_resolution_counts) > 1:
        formatted_sizes = ", ".join(summary["output_resolution_counts"])
        summary["folder_error"] = (
            "Images would have different output resolutions, which is not "
            f"valid for optical-flow training: {formatted_sizes}"
        )
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    if not output_resolution_counts:
        summary["folder_error"] = "No PNG images were found in the input folder."
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    output_size = next(iter(output_resolution_counts))
    try:
        updated_cameras_text = update_camera_intrinsics(cameras_text, output_size)
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
                "Existing output cameras.txt does not match this resize."
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

    for source_path in tqdm(
        image_files,
        desc="Preparing optical-flow images",
        unit="image",
    ):
        output_path = output_folder / source_path.name
        temporary_path: Path | None = None

        try:
            with Image.open(source_path) as source_image:
                source_image.load()
                source_size = source_image.size
                target_size = calculate_downscaled_size(*source_size, max_width)

                if output_path.exists():
                    with Image.open(output_path) as existing_image:
                        existing_image.load()
                        if existing_image.size != target_size:
                            raise RuntimeError(
                                "existing output dimension mismatch: "
                                f"expected={target_size}, output={existing_image.size}"
                            )
                        if existing_image.mode != "RGBA":
                            raise RuntimeError(
                                "existing output is not RGBA: "
                                f"mode={existing_image.mode!r}"
                            )
                    summary["skipped"] += 1
                    tqdm.write(f"Skipped verified existing output: {output_path}")
                    continue

                output_image = downscale_image_for_flow(source_image, max_width)
                temporary_path = output_folder / (
                    f".{source_path.stem}.{uuid.uuid4().hex}.tmp.png"
                )
                try:
                    output_image.save(temporary_path, format="PNG")
                finally:
                    output_image.close()

            with Image.open(temporary_path) as saved_image:
                saved_image.load()
                saved_size = saved_image.size
                saved_mode = saved_image.mode

            if saved_size != target_size:
                raise RuntimeError(
                    "saved output dimension mismatch: "
                    f"expected={target_size}, output={saved_size}"
                )
            if saved_mode != "RGBA":
                raise RuntimeError(
                    f"saved output is not RGBA: mode={saved_mode!r}"
                )

            temporary_path.rename(output_path)
            temporary_path = None
            summary["processed"] += 1
            if source_size == target_size:
                summary["already_within_limit"] += 1
            else:
                summary["resized"] += 1
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

    summary["elapsed_seconds"] = time.perf_counter() - started_at
    _print_summary(summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downscale PNG images and their alpha channels for RotGS "
            "optical-flow training."
        )
    )
    parser.add_argument("folder_name", help="Folder containing PNG images")
    parser.add_argument(
        "--cameras",
        required=True,
        help="input COLMAP cameras.txt file to scale with the images",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="maximum output width without upscaling (default: 1600)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = downscale_folder_for_flow(
            args.folder_name,
            args.cameras,
            args.max_width,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 1 if summary["folder_error"] or summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
