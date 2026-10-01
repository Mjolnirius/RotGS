"""Load physical turntable calibration produced by the TAE preprocessing step."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TAE_DIRECTORY_NAME = "turntable_angle_estimation"
TAE_AXIS_WIGGLE_DEG = 0.5
TAE_MIN_LOCAL_CORRECTION_DEG = 0.05
TAE_DEFAULT_SWEEP_CORRECTION_DEG = 0.15
TAE_CENTER_WIGGLE = 0.05
TAE_RESIDUAL_CONTROL_POINTS = 8


@dataclass(frozen=True)
class TAECalibration:
    directory: Path
    angles_by_image_deg: dict[str, float]
    uncertainties_by_image_deg: dict[str, float]
    axis_camera: tuple[float, float, float]
    axis_tilt_deg: float
    axis_side_deg: float
    rotation_center_camera_mm: tuple[float, float, float]
    maximum_angle_uncertainty_deg: float
    measured_total_rotation_deg: float


def configure_tae_training_args(
    args: object,
    dataset_folder: str | Path,
    explicit_options: set[str] | None = None,
) -> TAECalibration:
    """Apply physically anchored TAE defaults while honoring correction overrides."""
    explicit_options = explicit_options or set()
    calibration = load_tae_calibration(dataset_folder)

    if getattr(args, "multi_camera", False):
        raise ValueError("--leverage_TAE currently supports one fixed camera")
    if getattr(args, "angle_noise_std", 0.0) != 0.0:
        raise ValueError(
            "--angle_noise_std cannot be combined with --leverage_TAE"
        )

    args.axis_mode = "bounded_tilt"
    args.axis_tilt_init_deg = calibration.axis_tilt_deg
    args.axis_tilt_min_deg = max(0.0, calibration.axis_tilt_deg - TAE_AXIS_WIGGLE_DEG)
    args.axis_tilt_max_deg = min(180.0, calibration.axis_tilt_deg + TAE_AXIS_WIGGLE_DEG)
    args.axis_side_init_deg = calibration.axis_side_deg
    args.axis_side_limit_deg = TAE_AXIS_WIGGLE_DEG

    local_bound = max(
        TAE_MIN_LOCAL_CORRECTION_DEG,
        3.0 * calibration.maximum_angle_uncertainty_deg,
    )
    if "--max_residual_angle_deg" not in explicit_options:
        args.max_residual_angle_deg = local_bound
    if "--max_sweep_error_deg" not in explicit_options:
        args.max_sweep_error_deg = TAE_DEFAULT_SWEEP_CORRECTION_DEG
    if "--center_max_offset" not in explicit_options:
        args.center_max_offset = TAE_CENTER_WIGGLE
    if not getattr(args, "wo_tiny", False) and args.max_residual_angle_deg <= 0:
        raise ValueError(
            "TAE local corrections require --max_residual_angle_deg > 0; "
            "use --wo_tiny to disable them"
        )

    args.tae_directory = str(calibration.directory)
    args.tae_measured_total_rotation_deg = (
        calibration.measured_total_rotation_deg
    )
    args.tae_max_angle_uncertainty_deg = (
        calibration.maximum_angle_uncertainty_deg
    )
    args.tae_local_correction_bound_deg = (
        0.0 if getattr(args, "wo_tiny", False) else args.max_residual_angle_deg
    )
    args.tae_residual_control_points = TAE_RESIDUAL_CONTROL_POINTS
    return calibration


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TAE {label} is not a number: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"TAE {label} must be finite: {value!r}")
    return result


def _vector3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"TAE {label} must contain exactly three values")
    return tuple(
        _finite_float(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )


def load_tae_calibration(
    dataset_folder: str | Path,
    image_names: Iterable[str] | None = None,
) -> TAECalibration:
    """Load and strictly join TAE source rows to prepared RotGS image names."""
    dataset_path = Path(dataset_folder).expanduser().resolve()
    tae_directory = dataset_path / TAE_DIRECTORY_NAME
    csv_path = tae_directory / "turntable_angles.csv"
    summary_path = tae_directory / "summary.json"
    metadata_path = dataset_path / "preprocessing_metadata.json"

    for required_path in (csv_path, summary_path, metadata_path):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"--leverage_TAE requires {required_path}"
            )

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid TAE/preprocessing JSON: {exc}") from exc

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"TAE angle table is empty: {csv_path}")

    required_columns = {
        "frame_index",
        "filename",
        "nominal_angle_deg",
        "estimated_angle_deg",
        "angle_uncertainty_deg",
        "confidence",
    }
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        raise ValueError(
            "TAE angle table is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    try:
        rows.sort(key=lambda row: int(row["frame_index"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("TAE frame indices must be integers") from exc
    actual_indices = [int(row["frame_index"]) for row in rows]
    expected_indices = list(range(len(rows)))
    if actual_indices != expected_indices:
        raise ValueError(
            "TAE frame indices must be contiguous and start at zero; "
            f"found {actual_indices}"
        )

    renamed_images = metadata.get("renamed_images")
    if not isinstance(renamed_images, list) or len(renamed_images) != len(rows):
        raise ValueError(
            "preprocessing_metadata.json renamed_images does not match the "
            f"{len(rows)} TAE frames"
        )

    angles_by_image: dict[str, float] = {}
    uncertainties_by_image: dict[str, float] = {}
    estimated_angles: list[float] = []
    for index, (row, renamed) in enumerate(zip(rows, renamed_images)):
        if not isinstance(renamed, dict):
            raise ValueError(f"renamed_images[{index}] must be an object")
        source_name = Path(str(renamed.get("source", ""))).name
        tae_source_name = Path(str(row["filename"])).name
        if source_name != tae_source_name:
            raise ValueError(
                f"TAE frame {index} source {tae_source_name!r} does not match "
                f"preprocessing source {source_name!r}"
            )

        nominal = _finite_float(
            row["nominal_angle_deg"], f"frame {index} nominal angle"
        )
        metadata_nominal = _finite_float(
            renamed.get("angle_degrees"),
            f"preprocessing frame {index} nominal angle",
        )
        if not math.isclose(nominal, metadata_nominal, abs_tol=1e-6):
            raise ValueError(
                f"TAE frame {index} nominal angle {nominal} does not match "
                f"preprocessing angle {metadata_nominal}"
            )

        confidence = str(row["confidence"]).strip().lower()
        if confidence == "unresolved" or not str(row["estimated_angle_deg"]).strip():
            raise ValueError(f"TAE frame {index} has no resolved physical angle")
        estimated = _finite_float(
            row["estimated_angle_deg"], f"frame {index} estimated angle"
        )
        uncertainty = _finite_float(
            row["angle_uncertainty_deg"], f"frame {index} uncertainty"
        )
        if uncertainty < 0.0:
            raise ValueError(f"TAE frame {index} uncertainty must be nonnegative")

        output_name = Path(str(renamed.get("output", ""))).name
        if not output_name or output_name in angles_by_image:
            raise ValueError(
                f"invalid or duplicate preprocessing output for frame {index}: "
                f"{output_name!r}"
            )
        angles_by_image[output_name] = estimated
        uncertainties_by_image[output_name] = uncertainty
        estimated_angles.append(estimated)

    if any(
        current <= previous
        for previous, current in zip(estimated_angles, estimated_angles[1:])
    ):
        raise ValueError("TAE estimated angles must be strictly increasing")

    if image_names is not None:
        actual_names = set(image_names)
        expected_names = set(angles_by_image)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise ValueError(
                "TAE/preprocessed image mapping mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )

    try:
        summary_count = int(summary["input"]["image_count"])
        axis_value = summary["geometry"]["world_frame"][
            "basis_vectors_in_opencv_camera_coordinates"
        ]["z"]
        center_value = summary["geometry"]["rotation_centre_camera_mm"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("TAE summary is missing required geometry fields") from exc
    if summary_count != len(rows):
        raise ValueError(
            f"TAE summary reports {summary_count} images but CSV contains {len(rows)}"
        )

    axis = _vector3(axis_value, "camera-coordinate rotation axis")
    axis_norm = math.sqrt(sum(component * component for component in axis))
    if axis_norm <= 1e-8:
        raise ValueError("TAE camera-coordinate rotation axis is near zero")
    axis = tuple(component / axis_norm for component in axis)
    side_deg = math.degrees(math.asin(max(-1.0, min(1.0, axis[0]))))
    tilt_deg = math.degrees(math.atan2(axis[2], axis[1]))
    center = _vector3(center_value, "rotation centre in camera coordinates")
    if center[2] <= 0.0:
        raise ValueError("TAE rotation centre must lie in front of the camera")

    return TAECalibration(
        directory=tae_directory,
        angles_by_image_deg=angles_by_image,
        uncertainties_by_image_deg=uncertainties_by_image,
        axis_camera=axis,
        axis_tilt_deg=tilt_deg,
        axis_side_deg=side_deg,
        rotation_center_camera_mm=center,
        maximum_angle_uncertainty_deg=max(uncertainties_by_image.values()),
        measured_total_rotation_deg=estimated_angles[-1] - estimated_angles[0],
    )
