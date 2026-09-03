"""Interactively compare PNG RGB content with its alpha mask."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


PNG_EXTENSION = ".png"
IMAGES_DIRECTORY_NAME = "images"
WINDOW_NAME = "Alpha mask inspector"
PREVIOUS_KEYS = {
    81,  # OpenCV legacy left
    82,  # OpenCV legacy up
    63232,  # macOS up
    63234,  # macOS left
    65361,  # X11 left
    65362,  # X11 up
    2424832,  # Windows left
    2490368,  # Windows up
}
NEXT_KEYS = {
    83,  # OpenCV legacy right
    84,  # OpenCV legacy down
    63233,  # macOS down
    63235,  # macOS right
    65363,  # X11 right
    65364,  # X11 down
    2555904,  # Windows right
    2621440,  # Windows down
}


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required; run this utility through the project "
            "environment with 'uv run python'"
        ) from exc
    return cv2


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    """Sort numeric filenames such as 1.png, 2.png, 10.png naturally."""
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def find_alpha_pngs(folder_name: str | Path) -> tuple[Path, list[Path]]:
    """Return direct PNGs, falling back to a standard ``images`` subfolder."""
    folder = Path(folder_name).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError(f"input path is not a folder: {folder}")

    def png_files_in(candidate: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() == PNG_EXTENSION
            ),
            key=_natural_sort_key,
        )

    image_files = png_files_in(folder)
    if not image_files:
        images_folder = folder / IMAGES_DIRECTORY_NAME
        if images_folder.is_dir():
            folder = images_folder.resolve(strict=True)
            image_files = png_files_in(folder)

    if not image_files:
        raise ValueError(
            "no PNG files found directly inside the input folder or its "
            f"'{IMAGES_DIRECTORY_NAME}' subfolder: {folder}"
        )
    return folder, image_files


def _load_rgb_and_alpha(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one PNG as an RGB array and a three-channel alpha visualization."""
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError(f"file is not a PNG image: {path.name}")
            if "A" not in image.getbands() and "transparency" not in image.info:
                raise ValueError(f"PNG has no alpha channel: {path.name}")
            rgba = image.convert("RGBA")
            try:
                rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8).copy()
                alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8).copy()
            finally:
                rgba.close()
    except UnidentifiedImageError as exc:
        raise ValueError(f"could not read PNG image: {path.name}") from exc

    alpha_rgb = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    return rgb, alpha_rgb


def _build_display(
    path: Path,
    index: int,
    image_count: int,
    display_max_width: int,
    display_max_height: int,
) -> np.ndarray:
    """Build a size-limited BGR image containing RGB and alpha side by side."""
    cv2 = _import_cv2()
    rgb, alpha_rgb = _load_rgb_and_alpha(path)
    divider_width = max(2, round(rgb.shape[1] * 0.005))
    divider = np.full((rgb.shape[0], divider_width, 3), 96, dtype=np.uint8)
    comparison = np.concatenate((rgb, divider, alpha_rgb), axis=1)

    header_height = max(36, min(64, round(comparison.shape[0] * 0.08)))
    header = np.full(
        (header_height, comparison.shape[1], 3),
        32,
        dtype=np.uint8,
    )
    combined = np.concatenate((header, comparison), axis=0)
    label_scale = max(0.45, min(1.0, header_height / 55.0))
    cv2.putText(
        combined,
        f"RGB   |   ALPHA     {index + 1}/{image_count}: {path.name}",
        (12, max(24, round(header_height * 0.68))),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (240, 240, 240),
        max(1, round(label_scale * 2)),
        cv2.LINE_AA,
    )

    scale = min(
        1.0,
        display_max_width / combined.shape[1],
        display_max_height / combined.shape[0],
    )
    if scale < 1.0:
        display_size = (
            max(1, round(combined.shape[1] * scale)),
            max(1, round(combined.shape[0] * scale)),
        )
        combined = cv2.resize(combined, display_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)


def inspect_alpha_mask(
    folder_name: str | Path,
    *,
    display_max_width: int = 1600,
    display_max_height: int = 900,
    start_index: int = 0,
) -> None:
    """Open an interactive RGB/alpha viewer for a folder of PNG images.

    Left or Up shows the previous image, Right or Down shows the next image,
    and navigation wraps at both ends. Escape closes the window.
    """
    if display_max_width <= 0 or display_max_height <= 0:
        raise ValueError("display dimensions must be positive")

    folder, image_files = find_alpha_pngs(folder_name)
    if not 0 <= start_index < len(image_files):
        raise ValueError(
            f"start_index must be between 0 and {len(image_files) - 1}; "
            f"got {start_index}"
        )

    cv2 = _import_cv2()
    index = start_index
    print(f"Inspecting {len(image_files)} alpha PNGs in: {folder}")
    print("Left/Up=previous, Right/Down=next, Escape=close")

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            display = _build_display(
                image_files[index],
                index,
                len(image_files),
                display_max_width,
                display_max_height,
            )
            cv2.imshow(WINDOW_NAME, display)
            cv2.resizeWindow(WINDOW_NAME, display.shape[1], display.shape[0])
            key = cv2.waitKeyEx(0)

            if key == 27:
                break
            if key in PREVIOUS_KEYS:
                index = (index - 1) % len(image_files)
            elif key in NEXT_KEYS:
                index = (index + 1) % len(image_files)
    except cv2.error as exc:
        raise RuntimeError(
            "Could not open the alpha-mask window. Run from a desktop session "
            "with GUI access."
        ) from exc
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively inspect PNG RGB content beside its alpha channel."
        )
    )
    parser.add_argument("folder_name", help="folder directly containing alpha PNGs")
    parser.add_argument(
        "--display-max-width",
        type=int,
        default=1600,
        help="maximum window content width (default: 1600)",
    )
    parser.add_argument(
        "--display-max-height",
        type=int,
        default=900,
        help="maximum window content height (default: 900)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="zero-based image index shown first (default: 0)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        inspect_alpha_mask(
            args.folder_name,
            display_max_width=args.display_max_width,
            display_max_height=args.display_max_height,
            start_index=args.start_index,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
