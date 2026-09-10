"""Prepare multiple sequential single-camera turntable passes for RotGS.

Each source folder is processed with the same undistortion, ROI, alpha, crop,
resize, and camera-intrinsics pipeline as prepare_rotgs_sequence_png.py. Camera
passes are sorted by a rough elevation parsed from folder names such as
product_top_5d and bundled beneath one atomic top-level output folder.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__:
    from .prepare_rotgs_sequence import SelectionCancelled
    from .prepare_rotgs_sequence import SOURCE_ASPECT_OUTPUT_SUFFIX
    from .prepare_rotgs_sequence_png import (
        REGENERATED_ALPHA_SQUARE_OUTPUT_SUFFIX,
        SOURCE_ALPHA_SQUARE_OUTPUT_SUFFIX,
        UNDISTORTED_OUTPUT_MARKER,
        _pinhole_camera_entries,
        prepare_rotgs_sequence_png,
    )
    from .rotgs_web_selection import DEFAULT_WEB_PORT
else:
    from prepare_rotgs_sequence import SelectionCancelled
    from prepare_rotgs_sequence import SOURCE_ASPECT_OUTPUT_SUFFIX
    from prepare_rotgs_sequence_png import (
        REGENERATED_ALPHA_SQUARE_OUTPUT_SUFFIX,
        SOURCE_ALPHA_SQUARE_OUTPUT_SUFFIX,
        UNDISTORTED_OUTPUT_MARKER,
        _pinhole_camera_entries,
        prepare_rotgs_sequence_png,
    )
    from rotgs_web_selection import DEFAULT_WEB_PORT


CAMERA_ANGLE_PATTERN = re.compile(
    r"(?:^|[_\-\s])"
    r"(?P<angle>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"d(?:eg(?:rees?)?)?"
    r"(?=$|[_\-\s])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CameraPass:
    """One source sequence and its rough physical camera elevation."""

    source_folder: Path
    rough_elevation_degrees: float
    product_stem: str


def _format_angle(angle: float) -> str:
    if not math.isfinite(angle):
        raise ValueError("camera elevations must be finite")
    magnitude = format(abs(angle), ".8g").replace(".", "p")
    return f"{'m' if angle < 0 else ''}{magnitude}d"


def _parse_camera_angle(folder_name: str) -> tuple[float, str]:
    """Extract the final angle token and return it plus the base name."""
    matches = list(CAMERA_ANGLE_PATTERN.finditer(folder_name))
    if not matches:
        raise ValueError(
            "could not read a rough camera elevation from folder name "
            f"{folder_name!r}; expected a token such as '_5d'."
        )
    match = matches[-1]
    angle = float(match.group("angle"))
    stem = (folder_name[: match.start()] + folder_name[match.end() :]).strip(
        "_- "
    )
    stem = re.sub(r"[_\-\s]+", "_", stem)
    if not stem:
        raise ValueError(
            f"folder name {folder_name!r} has no product name outside its angle token"
        )
    return angle, stem


def _resolve_camera_passes(
    folder_names: Sequence[str | Path],
    camera_angles: Sequence[float] | None = None,
) -> list[CameraPass]:
    if len(folder_names) < 2:
        raise ValueError("multi-camera preprocessing requires at least two folders")
    if camera_angles is not None and len(camera_angles) != len(folder_names):
        raise ValueError(
            "--camera-angles must provide exactly one value per input folder"
        )

    resolved_paths: list[Path] = []
    for folder_name in folder_names:
        path = Path(folder_name).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"input path is not a folder: {path}")
        resolved_paths.append(path)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("each input camera folder must be unique")

    passes: list[CameraPass] = []
    for index, path in enumerate(resolved_paths):
        detected_angle, product_stem = _parse_camera_angle(path.name)
        angle = (
            float(camera_angles[index])
            if camera_angles is not None
            else detected_angle
        )
        if not math.isfinite(angle):
            raise ValueError("camera elevations must be finite")
        passes.append(
            CameraPass(
                source_folder=path,
                rough_elevation_degrees=angle,
                product_stem=product_stem,
            )
        )

    product_stems = {item.product_stem.casefold() for item in passes}
    if len(product_stems) != 1:
        detail = ", ".join(
            f"{item.source_folder.name} -> {item.product_stem!r}"
            for item in passes
        )
        raise ValueError(
            "input folders do not describe one common product after removing "
            f"their camera-angle tokens: {detail}"
        )

    passes.sort(
        key=lambda item: (
            item.rough_elevation_degrees,
            item.source_folder.name.casefold(),
            item.source_folder.name,
        )
    )
    for previous, current in zip(passes, passes[1:]):
        if math.isclose(
            previous.rough_elevation_degrees,
            current.rough_elevation_degrees,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "camera elevations must be unique; duplicate "
                f"{current.rough_elevation_degrees:g} degrees"
            )
    return passes


def _camera_directory_name(index: int, angle: float) -> str:
    return f"camera_{index:02d}_{_format_angle(angle)}"


def _default_output_name(
    camera_passes: Sequence[CameraPass],
    *,
    overwrite_alpha_mask: bool,
    square_crop: bool,
    output_sizes: Sequence[tuple[int, int]],
) -> str:
    product_stem = camera_passes[0].product_stem
    angles = "_".join(
        _format_angle(item.rough_elevation_degrees) for item in camera_passes
    )
    if square_crop:
        alpha_suffix = (
            REGENERATED_ALPHA_SQUARE_OUTPUT_SUFFIX
            if overwrite_alpha_mask
            else SOURCE_ALPHA_SQUARE_OUTPUT_SUFFIX
        )
    else:
        alpha_suffix = SOURCE_ASPECT_OUTPUT_SUFFIX
    if len(set(output_sizes)) == 1:
        width, height = output_sizes[0]
        resolution = str(width) if width == height else f"{width}x{height}"
    else:
        resolution = "mixedres"
    return (
        f"{product_stem}_multi_{angles}{UNDISTORTED_OUTPUT_MARKER}"
        f"{alpha_suffix}_{resolution}"
    )


def _generated_camera_matrix(camera_folder: Path) -> tuple[tuple[int, int], np.ndarray]:
    cameras_file = camera_folder / "sparse" / "0" / "cameras.txt"
    entries = _pinhole_camera_entries(cameras_file.read_text(encoding="utf-8"))
    if len(entries) != 1:
        raise ValueError(
            f"expected exactly one PINHOLE entry in {cameras_file}; "
            f"found {len(entries)}"
        )
    width, height, camera_matrix = entries[0]
    return (width, height), camera_matrix


def _same_intrinsics(camera_matrices: Sequence[np.ndarray]) -> bool:
    if not camera_matrices:
        return True
    reference = camera_matrices[0]
    return all(
        np.allclose(reference, candidate, rtol=1e-9, atol=1e-6)
        for candidate in camera_matrices[1:]
    )


def _validate_turntable_sequences(camera_metadata: Sequence[dict[str, Any]]) -> None:
    reference = camera_metadata[0]
    reference_angles = reference["angles_degrees"]
    reference_count = reference["image_count"]
    for index, metadata in enumerate(camera_metadata[1:], start=1):
        if (
            metadata["image_count"] != reference_count
            or metadata["angles_degrees"] != reference_angles
        ):
            raise ValueError(
                "all camera passes must contain the same turntable-angle sequence; "
                f"camera 0 has {reference_count} images and camera {index} has "
                f"{metadata['image_count']}"
            )


def prepare_rotgs_multi_sequence_png(
    folder_names: Sequence[str | Path],
    cameras_file: str | Path,
    target_width: int | None = None,
    *,
    calibration_file: str | Path | None = None,
    output_folder: str | Path | None = None,
    camera_angles: Sequence[float] | None = None,
    background_threshold: float = 12.0,
    segmentation_max_width: int = 1600,
    display_max_width: int = 1400,
    display_max_height: int = 900,
    web_port: int = DEFAULT_WEB_PORT,
    angle_token_from_right: int | None = None,
    square_crop: bool = True,
    overwrite_alpha_mask: bool = False,
) -> dict[str, Any]:
    """Prepare and atomically bundle multiple sequential camera-elevation passes."""
    started_at = time.perf_counter()
    camera_passes = _resolve_camera_passes(folder_names, camera_angles)
    input_cameras_file = Path(cameras_file).expanduser().resolve(strict=True)
    if not input_cameras_file.is_file():
        raise ValueError(f"cameras path is not a file: {input_cameras_file}")
    if target_width is not None and target_width <= 0:
        raise ValueError("target_width must be positive")

    source_parents = {item.source_folder.parent for item in camera_passes}
    if output_folder is None:
        if len(source_parents) != 1:
            raise ValueError(
                "input folders have different parents; pass --output explicitly"
            )
        output_parent = next(iter(source_parents))
        requested_output: Path | None = None
    else:
        requested_output = Path(output_folder).expanduser().resolve()
        output_parent = requested_output.parent
        if not output_parent.is_dir():
            raise ValueError(f"output parent does not exist: {output_parent}")
        if requested_output.exists():
            raise FileExistsError(
                f"output folder already exists; refusing to mix runs: {requested_output}"
            )
        for item in camera_passes:
            if requested_output == item.source_folder:
                raise ValueError("output folder must differ from every input folder")
            if item.source_folder in requested_output.parents:
                raise ValueError("output folder must not be inside an input folder")

    camera_summaries: list[dict[str, Any]] = []
    child_metadata: list[dict[str, Any]] = []
    camera_matrices: list[np.ndarray] = []
    output_sizes: list[tuple[int, int]] = []

    with tempfile.TemporaryDirectory(
        dir=output_parent,
        prefix=f".{camera_passes[0].product_stem}_multi_",
        suffix=".tmp",
    ) as temporary_directory:
        staging_root = Path(temporary_directory)
        for index, camera_pass in enumerate(camera_passes):
            camera_directory = _camera_directory_name(
                index,
                camera_pass.rough_elevation_degrees,
            )
            label = (
                f"Camera {index + 1}/{len(camera_passes)} · "
                f"rough elevation {camera_pass.rough_elevation_degrees:g}° · "
                f"{camera_pass.source_folder.name}"
            )
            print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
            summary = prepare_rotgs_sequence_png(
                camera_pass.source_folder,
                input_cameras_file,
                target_width=target_width,
                calibration_file=calibration_file,
                background_threshold=background_threshold,
                segmentation_max_width=segmentation_max_width,
                display_max_width=display_max_width,
                display_max_height=display_max_height,
                web_port=web_port,
                angle_token_from_right=angle_token_from_right,
                square_crop=square_crop,
                overwrite_alpha_mask=overwrite_alpha_mask,
                destination_folder=staging_root / camera_directory,
                review_label=label,
                print_summary=False,
            )
            camera_folder = staging_root / camera_directory
            metadata = json.loads(
                (camera_folder / "preprocessing_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            generated_size, camera_matrix = _generated_camera_matrix(camera_folder)
            metadata_size = tuple(int(value) for value in metadata["output_size"])
            if generated_size != metadata_size:
                raise RuntimeError(
                    f"camera dimensions disagree for {camera_directory}: "
                    f"cameras.txt={generated_size}, metadata={metadata_size}"
                )
            camera_summaries.append(summary)
            child_metadata.append(metadata)
            camera_matrices.append(camera_matrix)
            output_sizes.append(generated_size)
            print(
                f"Prepared {camera_directory}: {metadata['image_count']} images, "
                f"{generated_size[0]}x{generated_size[1]}",
                flush=True,
            )

        _validate_turntable_sequences(child_metadata)
        all_sizes_equal = len(set(output_sizes)) == 1
        all_intrinsics_equal = _same_intrinsics(camera_matrices)
        if not all_sizes_equal:
            print(
                "Warning: camera outputs have different resolutions. The new "
                "metadata preserves this, but the current legacy train_multi.py "
                "must be adapted before training.",
                flush=True,
            )
        if not all_intrinsics_equal:
            print(
                "Note: crop-adjusted camera intrinsics differ between passes and "
                "must be honored by the adapted multi-view trainer.",
                flush=True,
            )

        if requested_output is None:
            final_output = output_parent / _default_output_name(
                camera_passes,
                overwrite_alpha_mask=overwrite_alpha_mask,
                square_crop=square_crop,
                output_sizes=output_sizes,
            )
        else:
            final_output = requested_output
        if final_output.exists():
            raise FileExistsError(
                f"output folder already exists; refusing to mix runs: {final_output}"
            )

        cameras_payload: list[dict[str, Any]] = []
        for index, (camera_pass, metadata, camera_matrix, output_size) in enumerate(
            zip(camera_passes, child_metadata, camera_matrices, output_sizes)
        ):
            camera_directory = _camera_directory_name(
                index,
                camera_pass.rough_elevation_degrees,
            )
            cameras_payload.append(
                {
                    "camera_index": index,
                    "directory": camera_directory,
                    "rough_elevation_degrees": camera_pass.rough_elevation_degrees,
                    "source_folder": str(camera_pass.source_folder),
                    "image_count": metadata["image_count"],
                    "turntable_angles_degrees": metadata["angles_degrees"],
                    "source_size": metadata["source_size"],
                    "output_size": list(output_size),
                    "product_selection_left_top_right_bottom": metadata[
                        "product_selection_left_top_right_bottom"
                    ],
                    "crop_box_left_top_right_bottom": metadata[
                        "crop_box_left_top_right_bottom"
                    ],
                    "camera_matrix": camera_matrix.tolist(),
                    "preprocessing_metadata": (
                        f"{camera_directory}/preprocessing_metadata.json"
                    ),
                    "cameras_file": f"{camera_directory}/sparse/0/cameras.txt",
                }
            )

        multi_metadata: dict[str, Any] = {
            "format_version": 1,
            "dataset_type": "rotgs_multi_camera",
            "output_folder": str(final_output),
            "camera_count": len(camera_passes),
            "camera_order": "ascending_rough_elevation_degrees",
            "rough_camera_elevations_degrees": [
                item.rough_elevation_degrees for item in camera_passes
            ],
            "input_cameras_file": str(input_cameras_file),
            "calibration_file": child_metadata[0]["undistortion"][
                "calibration_file"
            ],
            "undistortion_applied": True,
            "alpha_mode": (
                "regenerated" if overwrite_alpha_mask else "preserve_source"
            ),
            "crop_mode": (
                "smallest_square" if square_crop else "source_aspect"
            ),
            "requested_target_width": target_width,
            "all_output_sizes_equal": all_sizes_equal,
            "all_crop_adjusted_intrinsics_equal": all_intrinsics_equal,
            "turntable_image_count_per_camera": child_metadata[0]["image_count"],
            "turntable_angles_degrees": child_metadata[0]["angles_degrees"],
            "trainer_handoff": {
                "camera_elevation_initialization": (
                    "rough_camera_elevations_degrees"
                ),
                "camera_directory_order_is_semantic": True,
                "use_each_camera_crop_adjusted_intrinsics": True,
                "legacy_train_multi_same_resolution_compatible": all_sizes_equal,
                "legacy_train_multi_single_rasterizer_compatible": (
                    all_sizes_equal and all_intrinsics_equal
                ),
            },
            "cameras": cameras_payload,
        }
        (staging_root / "multi_camera_metadata.json").write_text(
            json.dumps(multi_metadata, indent=2),
            encoding="utf-8",
        )
        staging_root.rename(final_output)

    for index, summary in enumerate(camera_summaries):
        camera_directory = _camera_directory_name(
            index,
            camera_passes[index].rough_elevation_degrees,
        )
        final_camera_folder = final_output / camera_directory
        summary.update(
            {
                "output_folder": str(final_camera_folder),
                "output_cameras_file": str(
                    final_camera_folder / "sparse" / "0" / "cameras.txt"
                ),
                "metadata_file": str(
                    final_camera_folder / "preprocessing_metadata.json"
                ),
            }
        )

    elapsed = time.perf_counter() - started_at
    summary = {
        "output_folder": str(final_output),
        "metadata_file": str(final_output / "multi_camera_metadata.json"),
        "camera_count": len(camera_passes),
        "rough_camera_elevations_degrees": [
            item.rough_elevation_degrees for item in camera_passes
        ],
        "output_sizes": [list(size) for size in output_sizes],
        "all_output_sizes_equal": len(set(output_sizes)) == 1,
        "all_crop_adjusted_intrinsics_equal": _same_intrinsics(camera_matrices),
        "camera_summaries": camera_summaries,
        "elapsed_seconds": elapsed,
    }
    print("\nRotGS multi-camera preparation summary")
    print(f"  Output folder:            {summary['output_folder']}")
    print(f"  Cameras prepared:         {summary['camera_count']}")
    print(
        "  Rough elevations:         "
        + ", ".join(
            f"{angle:g}°"
            for angle in summary["rough_camera_elevations_degrees"]
        )
    )
    print(
        "  Output resolutions:       "
        + ", ".join(f"{width}x{height}" for width, height in output_sizes)
    )
    print(
        "  Per-camera intrinsics:    "
        + (
            "identical"
            if summary["all_crop_adjusted_intrinsics_equal"]
            else "different (preserved in each camera folder)"
        )
    )
    print(f"  Metadata:                 {summary['metadata_file']}")
    print(f"  Elapsed time:             {elapsed:.3f} seconds")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare multiple sequential camera-elevation PNG turntable passes "
            "as one RotGS multi-camera dataset. Rough elevations are read from "
            "folder-name tokens such as '_5d'."
        )
    )
    parser.add_argument(
        "folder_names",
        nargs="+",
        help="two or more folders containing source PNG sequences",
    )
    parser.add_argument(
        "--cameras",
        required=True,
        help="shared source-resolution COLMAP cameras.txt file",
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
        "--output",
        type=Path,
        default=None,
        help=(
            "top-level output folder; omit to create an informative sibling "
            "folder after all ROI widths are known"
        ),
    )
    parser.add_argument(
        "--camera-angles",
        nargs="+",
        type=float,
        default=None,
        help=(
            "override rough elevations in input-folder order; folder angle "
            "tokens are still required to identify the common product stem"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "initial final width for every camera; each ROI page may adjust it, "
            "and omission preserves each selected crop's native resolution"
        ),
    )
    parser.add_argument(
        "--keep-source-aspect",
        action="store_true",
        help="use source-aspect crops instead of the default smallest square",
    )
    parser.add_argument(
        "--overwrite-alpha-mask",
        action="store_true",
        help="discard source alpha and regenerate it after undistortion/cropping",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=12.0,
        help="initial Lab background distance for regenerated alpha (default: 12)",
    )
    parser.add_argument(
        "--segmentation-max-width",
        type=int,
        default=1600,
        help="maximum mask-detection width (default: 1600)",
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
        help=f"local browser-review server port (default: {DEFAULT_WEB_PORT})",
    )
    parser.add_argument(
        "--angle-token-from-right",
        type=int,
        default=None,
        help=(
            "override the zero-based numeric filename token index from the right "
            "for every turntable sequence"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        prepare_rotgs_multi_sequence_png(
            args.folder_names,
            args.cameras,
            target_width=args.width,
            calibration_file=args.calibration,
            output_folder=args.output,
            camera_angles=args.camera_angles,
            background_threshold=args.background_threshold,
            segmentation_max_width=args.segmentation_max_width,
            display_max_width=args.display_max_width,
            display_max_height=args.display_max_height,
            web_port=args.web_port,
            angle_token_from_right=args.angle_token_from_right,
            square_crop=not args.keep_source_aspect,
            overwrite_alpha_mask=args.overwrite_alpha_mask,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        SelectionCancelled,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
