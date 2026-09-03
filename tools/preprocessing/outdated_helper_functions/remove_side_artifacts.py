"""Replace configurable left and right image regions with pure white pixels."""

from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _validate_percentages(left_percent: float, right_percent: float) -> None:
    """Validate side-removal percentages and raise a clear error if invalid."""
    if not math.isfinite(left_percent) or not math.isfinite(right_percent):
        raise ValueError("left_percent and right_percent must be finite numbers")
    if not 0 <= left_percent < 100:
        raise ValueError(
            f"left_percent must satisfy 0 <= left_percent < 100; got {left_percent}"
        )
    if not 0 <= right_percent < 100:
        raise ValueError(
            f"right_percent must satisfy 0 <= right_percent < 100; got {right_percent}"
        )
    if left_percent + right_percent >= 100:
        raise ValueError(
            "left_percent + right_percent must be less than 100; "
            f"got {left_percent + right_percent}"
        )


def calculate_side_cutoffs(
    width: int,
    left_percent: float,
    right_percent: float,
) -> tuple[int, int]:
    """Return the exclusive left cutoff and inclusive right-region start."""
    _validate_percentages(left_percent, right_percent)
    if width <= 0:
        raise ValueError(f"image width must be positive; got {width}")

    left_cutoff = int(width * (left_percent / 100.0))
    right_cutoff = int(width * (1.0 - right_percent / 100.0))
    return left_cutoff, right_cutoff


def remove_side_artifacts(
    image: Image.Image,
    left_percent: float,
    right_percent: float,
) -> Image.Image:
    """Return a copy with the requested side regions replaced by pure white.

    RGB and RGBA modes are preserved. Other modes are converted to RGB, or to
    RGBA when the source contains transparency. The returned image retains the
    source dimensions and the center region is not cropped or resized.
    """
    left_cutoff, right_cutoff = calculate_side_cutoffs(
        image.width,
        left_percent,
        right_percent,
    )

    if image.mode in {"RGB", "RGBA"}:
        result = image.copy()
    else:
        has_transparency = "A" in image.getbands() or "transparency" in image.info
        result = image.convert("RGBA" if has_transparency else "RGB")

    white = (255, 255, 255, 255) if result.mode == "RGBA" else (255, 255, 255)
    if left_cutoff > 0:
        result.paste(white, (0, 0, left_cutoff, result.height))
    if right_cutoff < result.width:
        result.paste(white, (right_cutoff, 0, result.width, result.height))
    return result


def _new_summary(
    input_folder: Path,
    output_folder: Path,
    left_percent: float,
    right_percent: float,
) -> dict[str, Any]:
    return {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "left_percent": left_percent,
        "right_percent": right_percent,
        "found": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failures": [],
        "representative_cutoffs": None,
        "elapsed_seconds": 0.0,
        "folder_error": None,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nSide-artifact removal summary")
    print(f"  Input folder:             {summary['input_folder']}")
    print(f"  Output folder:            {summary['output_folder']}")
    print(f"  Left percentage:          {summary['left_percent']}")
    print(f"  Right percentage:         {summary['right_percent']}")
    print(f"  Images found:             {summary['found']}")
    print(f"  Successfully processed:   {summary['processed']}")
    print(f"  Skipped (already exists): {summary['skipped']}")
    print(f"  Failed:                   {summary['failed']}")
    print(f"  Elapsed time:             {summary['elapsed_seconds']:.3f} seconds")

    representative = summary["representative_cutoffs"]
    if representative is None:
        print("  Representative cutoffs:   none")
    else:
        print(
            "  Representative cutoffs:   "
            f"{representative['filename']} (width={representative['width']}, "
            f"left={representative['left_cutoff']}, "
            f"right={representative['right_cutoff']})"
        )

    if summary["failures"]:
        print("  Failures:")
        for failure in summary["failures"]:
            print(f"    {failure}")
    if summary["folder_error"]:
        print(f"  Folder error:             {summary['folder_error']}")


def process_folder(
    folder_name: str | Path,
    left_percent: float,
    right_percent: float,
) -> dict[str, Any]:
    """Process supported direct-child images into a sibling output folder."""
    _validate_percentages(left_percent, right_percent)
    started_at = time.perf_counter()
    input_folder = Path(folder_name).expanduser()
    output_folder = input_folder.with_name(f"{input_folder.name}_side_cropped")

    try:
        input_folder = input_folder.resolve(strict=True)
        output_folder = input_folder.with_name(f"{input_folder.name}_side_cropped")
    except (FileNotFoundError, OSError) as exc:
        summary = _new_summary(
            input_folder,
            output_folder,
            left_percent,
            right_percent,
        )
        summary["folder_error"] = (
            f"Input folder does not exist or is inaccessible: {exc}"
        )
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    summary = _new_summary(
        input_folder,
        output_folder,
        left_percent,
        right_percent,
    )
    if not input_folder.is_dir():
        summary["folder_error"] = "Input path exists but is not a folder."
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    try:
        output_folder.mkdir(exist_ok=True)
        image_files = sorted(
            (
                path
                for path in input_folder.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    except OSError as exc:
        summary["folder_error"] = f"Could not prepare folders: {exc}"
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        print(f"Error: {summary['folder_error']}", file=sys.stderr)
        _print_summary(summary)
        return summary

    summary["found"] = len(image_files)
    for source_path in tqdm(
        image_files,
        desc="Removing side artifacts",
        unit="image",
    ):
        output_path = output_folder / source_path.name
        temporary_path: Path | None = None

        try:
            with Image.open(source_path) as source_image:
                source_image.load()
                input_dimensions = source_image.size
                left_cutoff, right_cutoff = calculate_side_cutoffs(
                    source_image.width,
                    left_percent,
                    right_percent,
                )

                if summary["representative_cutoffs"] is None:
                    summary["representative_cutoffs"] = {
                        "filename": source_path.name,
                        "width": source_image.width,
                        "left_cutoff": left_cutoff,
                        "right_cutoff": right_cutoff,
                    }

                if output_path.exists():
                    summary["skipped"] += 1
                    tqdm.write(f"Skipped existing output: {output_path}")
                    continue

                processed_image = remove_side_artifacts(
                    source_image,
                    left_percent,
                    right_percent,
                )
                temporary_path = output_folder / (
                    f".{source_path.stem}.{uuid.uuid4().hex}.tmp{source_path.suffix}"
                )
                try:
                    processed_image.save(temporary_path)
                finally:
                    processed_image.close()

            with Image.open(temporary_path) as output_image:
                output_image.load()
                output_dimensions = output_image.size

            if output_dimensions != input_dimensions:
                raise RuntimeError(
                    "dimension mismatch: "
                    f"input={input_dimensions}, output={output_dimensions}"
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

    summary["elapsed_seconds"] = time.perf_counter() - started_at
    _print_summary(summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace configurable left and right regions of JPG/JPEG/PNG "
            "images with pure white pixels."
        )
    )
    parser.add_argument("folder_name", help="Folder containing input images")
    parser.add_argument(
        "--left",
        type=float,
        required=True,
        help="percentage of image width to replace from the left edge",
    )
    parser.add_argument(
        "--right",
        type=float,
        required=True,
        help="percentage of image width to replace from the right edge",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = process_folder(args.folder_name, args.left, args.right)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 1 if summary["folder_error"] or summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
