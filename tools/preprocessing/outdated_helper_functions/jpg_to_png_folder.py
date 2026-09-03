"""Convert the JPEG files in one folder into PNG files in a sibling folder."""

from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


JPEG_EXTENSIONS = {".jpg", ".jpeg"}
PNG_COMPATIBLE_MODES = {"1", "L", "LA", "I", "I;16", "P", "RGB", "RGBA"}
ANGLE_SUFFIX_PATTERN = re.compile(
    r"(?:^|_)(?P<angle>-?\d+(?:\.\d+)?)(?:_fake)?$",
    re.IGNORECASE,
)


def _format_bytes(number_of_bytes: int) -> str:
    """Return a compact human-readable byte count while retaining exact bytes."""
    size = float(number_of_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    return f"{number_of_bytes} bytes ({size:.2f} {unit})"


def _new_summary(
    input_folder: Path,
    output_folder: Path,
    rename_for_training: bool,
    convert_to_rgba: bool,
) -> dict[str, Any]:
    return {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "rename_for_training": rename_for_training,
        "convert_to_rgba": convert_to_rgba,
        "found": 0,
        "converted": 0,
        "failed": 0,
        "skipped": 0,
        "elapsed_seconds": 0.0,
        "average_seconds_per_image": 0.0,
        "total_jpg_bytes": 0,
        "converted_jpg_bytes": 0,
        "total_png_bytes": 0,
        "size_ratio": None,
        "resolution_counts": {},
        "color_mode_conversions": 0,
        "failures": [],
        "folder_error": None,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nConversion summary")
    print(f"  Input folder:             {summary['input_folder']}")
    print(f"  Output folder:            {summary['output_folder']}")
    print(
        "  Output color mode:        "
        + ("RGBA" if summary["convert_to_rgba"] else "preserved when possible")
    )
    print(
        "  Output naming:            "
        + (
            "training indices (0000.png, 0001.png, ...)"
            if summary["rename_for_training"]
            else "original filenames"
        )
    )
    print(f"  JPG files found:          {summary['found']}")
    print(f"  Converted:                {summary['converted']}")
    print(f"  Failed:                   {summary['failed']}")
    print(f"  Skipped (already exists): {summary['skipped']}")
    print(f"  Elapsed time:             {summary['elapsed_seconds']:.3f} seconds")
    print(
        "  Average seconds/image:   "
        f"{summary['average_seconds_per_image']:.4f}"
    )
    print(f"  Total JPG size:           {_format_bytes(summary['total_jpg_bytes'])}")
    print(f"  Total PNG size:           {_format_bytes(summary['total_png_bytes'])}")

    if summary["size_ratio"] is None:
        print("  PNG/JPG size ratio:       n/a (no images converted)")
    else:
        print(
            "  PNG/JPG size ratio:       "
            f"{summary['size_ratio']:.3f} "
            "(converted files only)"
        )

    print("  Image resolutions:")
    if summary["resolution_counts"]:
        for resolution, count in summary["resolution_counts"].items():
            print(f"    {resolution}: {count}")
    else:
        print("    none")

    if summary["color_mode_conversions"]:
        print(
            "  Color-mode conversions:  "
            f"{summary['color_mode_conversions']}"
        )

    if summary["folder_error"]:
        print(f"  Folder error:             {summary['folder_error']}")


def _angle_from_filename(path: Path) -> float:
    """Extract the trailing angle, optionally followed by ``_fake``."""
    match = ANGLE_SUFFIX_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(
            "filename must end with an angle or an angle followed by '_fake' "
            f"when training renaming is enabled: {path.name!r}"
        )
    return float(match.group("angle"))


def _output_folder_for(input_folder: Path, convert_to_rgba: bool) -> Path:
    suffix = "_PNGa" if convert_to_rgba else "_PNG"
    return input_folder.with_name(f"{input_folder.name}{suffix}")


def jpg_to_png_folder(
    folder_name: str | Path,
    rename_for_training: bool = True,
    convert_to_rgba: bool = True,
) -> dict[str, Any]:
    """Convert direct-child JPEGs to PNGs in a sibling output folder.

    Only files whose suffix is ``.jpg`` or ``.jpeg`` (case-insensitive) are
    processed. Images are decoded and saved without cropping, resizing, or
    applying EXIF orientation. Every written PNG is reopened, fully decoded,
    and checked against the input pixel dimensions and requested color mode.

    By default, outputs are converted to RGBA and written to
    ``<folder_name>_PNGa`` so the RotGS image loader can consume them. Because
    JPEG has no transparency, the added alpha channel is fully opaque. Set
    ``convert_to_rgba=False`` to preserve PNG-compatible source modes and use
    the legacy ``<folder_name>_PNG`` output folder.

    By default, the trailing angle is extracted from every filename, the files
    are sorted by that angle, and outputs receive zero-padded sequential names
    such as ``0000.png`` and ``0072.png``. A trailing ``_fake`` marker is
    accepted, so ``object_1_360_fake.jpg`` sorts at 360 degrees. Set
    ``rename_for_training=False`` to retain each source filename stem.

    Existing output PNGs are left untouched. If multiple JPEGs would produce
    the same PNG name when original names are retained (for example,
    ``photo.jpg`` and ``photo.jpeg``), each colliding input is reported as a
    failure instead of overwriting data.

    Returns a dictionary containing the printed summary statistics.
    """
    started_at = time.perf_counter()
    input_folder = Path(folder_name).expanduser()

    try:
        input_folder = input_folder.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        output_folder = _output_folder_for(input_folder, convert_to_rgba)
        summary = _new_summary(
            input_folder,
            output_folder,
            rename_for_training,
            convert_to_rgba,
        )
        summary["folder_error"] = f"Input folder does not exist or is inaccessible: {exc}"
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    output_folder = _output_folder_for(input_folder, convert_to_rgba)
    summary = _new_summary(
        input_folder,
        output_folder,
        rename_for_training,
        convert_to_rgba,
    )

    if not input_folder.is_dir():
        summary["folder_error"] = "Input path exists but is not a folder."
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

    try:
        jpg_files = sorted(
            (
                path
                for path in input_folder.iterdir()
                if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    except OSError as exc:
        summary["folder_error"] = f"Could not inspect input folder: {exc}"
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary
    summary["found"] = len(jpg_files)

    source_sizes: dict[Path, int] = {}
    for source_path in jpg_files:
        try:
            source_sizes[source_path] = source_path.stat().st_size
        except OSError:
            source_sizes[source_path] = 0
    summary["total_jpg_bytes"] = sum(source_sizes.values())

    if not jpg_files:
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"No JPG/JPEG files found in: {input_folder}")
        _print_summary(summary)
        return summary

    if rename_for_training:
        try:
            jpg_files.sort(
                key=lambda path: (
                    _angle_from_filename(path),
                    path.name.casefold(),
                    path.name,
                )
            )
        except ValueError as exc:
            summary["folder_error"] = str(exc)
            summary["elapsed_seconds"] = time.perf_counter() - started_at
            print(f"Error: {summary['folder_error']}", file=sys.stderr)
            _print_summary(summary)
            return summary

        index_width = max(4, len(str(len(jpg_files) - 1)))
        output_names = {
            path: f"{index:0{index_width}d}.png"
            for index, path in enumerate(jpg_files)
        }
    else:
        output_names = {path: f"{path.stem}.png" for path in jpg_files}

    output_name_counts = Counter(name.casefold() for name in output_names.values())
    resolution_counts: Counter[tuple[int, int]] = Counter()

    for source_path in tqdm(jpg_files, desc="Converting JPG to PNG", unit="image"):
        output_path = output_folder / output_names[source_path]
        temporary_path: Path | None = None

        try:
            with Image.open(source_path) as source_image:
                source_image.load()
                input_dimensions = source_image.size
                resolution_counts[input_dimensions] += 1

                if output_name_counts[output_path.name.casefold()] > 1:
                    raise RuntimeError(
                        f"multiple input JPEGs map to the output name {output_path.name!r}"
                    )

                if output_path.exists():
                    with Image.open(output_path) as existing_image:
                        existing_image.load()
                        if existing_image.size != input_dimensions:
                            raise RuntimeError(
                                "existing output dimension mismatch: "
                                f"input={input_dimensions}, output={existing_image.size}"
                            )
                        if convert_to_rgba and existing_image.mode != "RGBA":
                            raise RuntimeError(
                                "existing output is not RGBA: "
                                f"mode={existing_image.mode!r}"
                            )
                    summary["skipped"] += 1
                    tqdm.write(f"Skipped verified existing output: {output_path}")
                    continue

                image_to_save = source_image
                converted_image: Image.Image | None = None
                if convert_to_rgba and source_image.mode != "RGBA":
                    converted_image = source_image.convert("RGBA")
                    image_to_save = converted_image
                    summary["color_mode_conversions"] += 1
                elif source_image.mode not in PNG_COMPATIBLE_MODES:
                    converted_image = source_image.convert("RGB")
                    image_to_save = converted_image
                    summary["color_mode_conversions"] += 1

                temporary_path = output_folder / (
                    f".{source_path.stem}.{uuid.uuid4().hex}.tmp.png"
                )
                try:
                    image_to_save.save(temporary_path, format="PNG")
                finally:
                    if converted_image is not None:
                        converted_image.close()

            with Image.open(temporary_path) as output_image:
                output_image.load()
                output_dimensions = output_image.size
                output_mode = output_image.mode

            if output_dimensions != input_dimensions:
                raise RuntimeError(
                    "dimension mismatch: "
                    f"input={input_dimensions}, output={output_dimensions}"
                )
            if convert_to_rgba and output_mode != "RGBA":
                raise RuntimeError(
                    f"output color-mode mismatch: expected='RGBA', output={output_mode!r}"
                )

            temporary_path.rename(output_path)
            temporary_path = None
            summary["converted"] += 1
            summary["converted_jpg_bytes"] += source_sizes[source_path]
            summary["total_png_bytes"] += output_path.stat().st_size

        except Exception as exc:  # Continue converting independent images.
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
    summary["average_seconds_per_image"] = (
        summary["elapsed_seconds"] / summary["found"] if summary["found"] else 0.0
    )
    summary["size_ratio"] = (
        summary["total_png_bytes"] / summary["converted_jpg_bytes"]
        if summary["converted_jpg_bytes"]
        else None
    )
    summary["resolution_counts"] = {
        f"{width}x{height}": count
        for (width, height), count in sorted(resolution_counts.items())
    }

    _print_summary(summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert JPG/JPEG files into PNG files in a sibling _PNGa folder "
            "(or _PNG when RGBA conversion is disabled)."
        )
    )
    parser.add_argument("folder_name", help="Folder containing JPG/JPEG files")
    parser.add_argument(
        "--rename-for-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "sort files by the trailing angle and name outputs 0000.png, "
            "0001.png, ... (default: enabled)"
        ),
    )
    parser.add_argument(
        "--rgba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "convert outputs to RGBA and use an _PNGa folder "
            "(default: enabled)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = jpg_to_png_folder(
        args.folder_name,
        rename_for_training=args.rename_for_training,
        convert_to_rgba=args.rgba,
    )
    return 1 if summary["folder_error"] or summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
