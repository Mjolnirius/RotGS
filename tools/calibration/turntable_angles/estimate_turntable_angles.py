#!/usr/bin/env python3
"""Estimate true physical turntable angles from calibrated ArUco markers."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .turntable_geometry import (
    EstimationError,
    detect_frames,
    discover_images,
    estimate_turntable_geometry,
    load_calibration,
)
from .turntable_reports import write_all_outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the physical rotation angle of a fixed-camera turntable "
            "sequence from unique, coplanar ArUco markers. The marker layout, "
            "fixed 3D rotation axis, and independent per-frame angles are refined "
            "globally from calibrated image-space corner residuals."
        ),
        epilog=(
            "The intrinsics file must be the camera_calibration.npz produced by "
            "tools/calibration/get_camera_intrinsics.py. Image dimensions must "
            "match it exactly; this tool never scales intrinsics implicitly."
        ),
    )
    parser.add_argument(
        "image_folder",
        metavar="IMAGE_FOLDER",
        type=Path,
        help="folder containing the turntable JPG, JPEG, or PNG sequence",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        required=True,
        help="camera_calibration.npz containing K, distortion, and image_size",
    )
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        required=True,
        help=(
            "physical side length in millimetres of the BLACK ArUco square "
            "(not the outer sticker or paper)"
        ),
    )
    parser.add_argument(
        "--nominal-step-deg",
        type=float,
        default=5.0,
        help=(
            "nominal acquisition step used only for initialization, unwrapping, "
            "and error reporting (default: 5.0)"
        ),
    )
    parser.add_argument(
        "--start-angle-deg",
        type=float,
        default=0.0,
        help=(
            "start of the nominal reporting sequence; measured angles remain "
            "relative to frame 0 (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--aruco-dictionary",
        default="DICT_5X5_50",
        help="OpenCV predefined ArUco dictionary name (default: DICT_5X5_50)",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help=(
            "optional empty-turntable image; its angle is estimated as an "
            "auxiliary variable, so it need not be captured at frame 0"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "output directory (default: IMAGE_FOLDER/turntable_angle_estimation)"
        ),
    )
    parser.add_argument(
        "--no-annotated-images",
        action="store_true",
        help="skip writing one diagnostic annotated image per source frame",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print detailed detection and SciPy optimization progress",
    )
    return parser.parse_args(argv)


def _require_finite(name: str, value: float, positive: bool = False) -> None:
    if not math.isfinite(value) or (positive and value <= 0.0):
        requirement = "a positive finite value" if positive else "finite"
        raise ValueError(f"{name} must be {requirement}; got {value!r}")


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "unresolved"
    return f"{value:.{digits}f}"


def _print_summary(
    summary: dict[str, Any], outputs: dict[str, Any], elapsed_seconds: float
) -> None:
    inputs = summary["input"]
    angles = summary["angle_results"]
    geometry = summary["geometry"]
    camera = summary["camera"]
    optimization = summary["optimization"]
    print()
    print("=" * 60)
    print("TURNTABLE ANGLE ESTIMATION")
    print("=" * 60)
    print(f"Images                         : {inputs['image_count']}")
    print(
        "Unique ArUco markers           : "
        f"{len(optimization['unique_detected_marker_ids'])}"
    )
    print(f"Marker size                    : {inputs['marker_size_mm']:.3f} mm")
    print(f"Nominal step                   : {inputs['nominal_step_deg']:.4f} deg")
    print(
        "Expected total rotation        : "
        f"{inputs['expected_total_rotation_deg']:.4f} deg"
    )
    print("\n## ANGLE RESULTS\n")
    print(
        f"Mean measured step             : {_format(angles['mean_step_deg'])} deg"
    )
    print(
        f"Median measured step           : {_format(angles['median_step_deg'])} deg"
    )
    print(
        "Std. deviation                 : "
        f"{_format(angles['step_standard_deviation_deg'])} deg"
    )
    print(
        f"RMS step error                 : {_format(angles['rms_step_error_deg'])} deg"
    )
    print(
        "Maximum absolute step error    : "
        f"{_format(angles['maximum_absolute_step_error_deg'])} deg"
    )
    print(
        "Measured total rotation        : "
        f"{_format(angles['measured_total_rotation_deg'])} deg"
    )
    print(
        f"Closure error                  : {_format(angles['closure_error_deg'])} deg"
    )
    if angles["unresolved_frame_indices"]:
        print(
            "Unresolved source frames       : "
            + ", ".join(map(str, angles["unresolved_frame_indices"]))
        )
    print("\n## GEOMETRY\n")
    print(
        "Rotation axis                  : "
        f"{geometry['rotation_axis_direction_world']}"
    )
    print(
        "Reference table normal (theta0): "
        f"{geometry['reference_table_normal_theta_0_world']}"
    )
    print(
        "Axis <-> reference normal      : "
        f"{geometry['axis_vs_reference_table_normal_deg']:.4f} deg"
    )
    print(
        "Rotation centre                : "
        f"{geometry['rotation_centre_world_mm']} mm (world origin)"
    )
    print("\n## CAMERA\n")
    print(f"Position                       : {camera['position_world_mm']} mm")
    print(
        "Camera-to-axis distance        : "
        f"{camera['camera_centre_to_rotation_axis_distance_mm']:.3f} mm"
    )
    print(
        "Optical axis <-> rotation axis : "
        f"{camera['optical_axis_vs_rotation_axis_deg']:.4f} deg"
    )
    print(
        "Downward tilt vs World XY      : "
        f"{camera['downward_tilt_relative_world_xy_deg']:.4f} deg"
    )
    print(
        "Downward tilt vs reference plane: "
        f"{camera['downward_tilt_relative_reference_table_plane_theta_0_deg']:.4f} deg"
    )
    print("\n## FIT QUALITY\n")
    print(
        f"Detected observations          : {optimization['detected_observations']}"
    )
    print(f"Used observations              : {optimization['used_observations']}")
    print(
        f"Rejected observations          : {optimization['rejected_observations']}"
    )
    print(
        "Mean reprojection error        : "
        f"{_format(optimization['final_mean_reprojection_px'])} px"
    )
    print(
        "RMS reprojection error         : "
        f"{_format(optimization['final_rms_reprojection_px'])} px"
    )
    print(
        "Maximum reprojection error     : "
        f"{_format(optimization['final_max_reprojection_px'])} px"
    )
    print(
        "Optimization                   : "
        + ("converged" if optimization["optimizer_success"] else "failed")
    )
    print(f"Elapsed                        : {elapsed_seconds:.1f} s")
    print("\n## OUTPUT\n")
    print(f"Angles CSV                     : {outputs['csv']}")
    print(f"Summary                        : {outputs['summary']}")
    print(f"Plots                          : {outputs['csv'].parent}")
    print(
        "Annotated images               : "
        + (str(outputs["annotated"]) if outputs["annotated"] else "skipped")
    )


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    _require_finite("--marker-size-mm", args.marker_size_mm, positive=True)
    _require_finite("--nominal-step-deg", args.nominal_step_deg, positive=True)
    _require_finite("--start-angle-deg", args.start_angle_deg)
    images = discover_images(args.image_folder)
    calibration = load_calibration(args.intrinsics)
    image_folder = args.image_folder.expanduser().resolve(strict=True)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else image_folder / "turntable_angle_estimation"
    )
    if output_dir == image_folder:
        raise ValueError(
            "--output-dir must not be the source image folder because generated "
            "PNG plots would become sequence inputs on the next run"
        )
    print("Detecting calibrated ArUco corners...", flush=True)
    frames, reference = detect_frames(
        images,
        calibration,
        args.marker_size_mm,
        args.aruco_dictionary,
        args.reference_image,
        args.verbose,
    )
    print("Initializing the rigid marker graph and fixed rotation axis...", flush=True)
    result = estimate_turntable_geometry(
        frames,
        reference,
        calibration,
        args.marker_size_mm,
        args.aruco_dictionary,
        args.nominal_step_deg,
        args.verbose,
    )
    print("Writing measurements and diagnostics...", flush=True)
    _, summary, outputs = write_all_outputs(
        output_dir,
        result,
        args.nominal_step_deg,
        args.start_angle_deg,
        not args.no_annotated_images,
    )
    _print_summary(summary, outputs, time.perf_counter() - started)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run(args)
    except (
        EstimationError,
        FileNotFoundError,
        OSError,
        ValueError,
        np.linalg.LinAlgError,
        cv2.error,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
