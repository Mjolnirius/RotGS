"""Machine-readable and visual reports for turntable-angle estimation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .turntable_geometry import (
    EstimationResult,
    FrameRecord,
    MarkerObservation,
    project_camera_points,
)


CONFIDENCE_RULES = {
    "unresolved": (
        "no globally accepted marker observation or no measured angle"
    ),
    "excellent": (
        "at least 3 used markers/12 corners, corner-hull coverage >=2% of the "
        "image, RMS reprojection <=0.75 px, approximate 1-sigma angle uncertainty "
        "<=0.05 deg, rejection fraction <=25%, and no neighbour-consistency warning"
    ),
    "good": (
        "at least 2 used markers/8 corners, corner-hull coverage >=0.5% of the "
        "image, RMS reprojection <=1.5 px, approximate 1-sigma angle uncertainty "
        "<=0.15 deg, rejection fraction <=50%, and no neighbour-consistency warning"
    ),
    "low": "a solved angle that does not satisfy every good/excellent condition",
}


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1)]


def _used_observations(frame: FrameRecord) -> list[MarkerObservation]:
    return [item for item in frame.observations if item.used]


def _coverage(frame: FrameRecord) -> float:
    used = _used_observations(frame)
    if not used:
        return 0.0
    points = np.vstack([item.corners for item in used]).astype(np.float32)
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points)
    area = float(cv2.contourArea(hull))
    return area / float(frame.image_size[0] * frame.image_size[1])


def _frame_errors(frame: FrameRecord) -> np.ndarray:
    values = [
        observation.reprojection_errors
        for observation in frame.observations
        if observation.used and observation.reprojection_errors is not None
    ]
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def _neighbour_warning(
    index: int,
    measured: list[float | None],
    uncertainties: list[float | None],
) -> bool:
    if index <= 0 or index >= len(measured) - 1:
        return False
    before, current, after = measured[index - 1 : index + 2]
    if before is None or current is None or after is None:
        return False
    departure = abs(current - 0.5 * (before + after))
    uncertainty = uncertainties[index]
    tolerance = max(1.0, 5.0 * uncertainty) if uncertainty is not None else 1.0
    return departure > tolerance


def build_frame_rows(
    result: EstimationResult,
    nominal_step_deg: float,
    start_angle_deg: float,
) -> list[dict[str, Any]]:
    measured: list[float | None] = []
    uncertainties: list[float | None] = []
    for frame in result.frames:
        if frame.index in result.unresolved_frames:
            measured.append(None)
            uncertainties.append(None)
        else:
            measured.append(math.degrees(result.state.theta[frame.index]))
            uncertainties.append(
                _finite_or_none(result.angle_uncertainty_deg.get(frame.index))
            )

    rows: list[dict[str, Any]] = []
    for frame in result.frames:
        index = frame.index
        nominal = start_angle_deg + index * nominal_step_deg
        angle = measured[index]
        previous = measured[index - 1] if index > 0 else None
        step = angle - previous if angle is not None and previous is not None else None
        step_error = step - nominal_step_deg if step is not None else None
        cumulative = angle - nominal if angle is not None else None
        used = _used_observations(frame)
        errors = _frame_errors(frame)
        coverage = _coverage(frame)
        uncertainty = uncertainties[index]
        rejection_fraction = (
            (len(frame.observations) - len(used)) / len(frame.observations)
            if frame.observations
            else 1.0
        )
        neighbour_warning = _neighbour_warning(index, measured, uncertainties)
        rms = float(np.sqrt(np.mean(errors * errors))) if errors.size else None
        if angle is None or not used:
            confidence = "unresolved"
        elif (
            len(used) >= 3
            and len(used) * 4 >= 12
            and coverage >= 0.02
            and rms is not None
            and rms <= 0.75
            and uncertainty is not None
            and uncertainty <= 0.05
            and rejection_fraction <= 0.25
            and not neighbour_warning
        ):
            confidence = "excellent"
        elif (
            len(used) >= 2
            and len(used) * 4 >= 8
            and coverage >= 0.005
            and rms is not None
            and rms <= 1.5
            and uncertainty is not None
            and uncertainty <= 0.15
            and rejection_fraction <= 0.5
            and not neighbour_warning
        ):
            confidence = "good"
        else:
            confidence = "low"
        rows.append(
            {
                "frame_index": index,
                "filename": frame.filename,
                "nominal_angle_deg": nominal,
                "estimated_angle_deg": angle,
                "step_deg": step,
                "step_error_deg": step_error,
                "cumulative_error_deg": cumulative,
                "angle_uncertainty_deg": uncertainty,
                "detected_marker_count": len(frame.observations),
                "used_marker_count": len(used),
                "used_corner_count": len(used) * 4,
                "rejected_observation_count": len(frame.observations) - len(used),
                "detector_rejected_candidate_count": frame.rejected_detection_count,
                "corner_hull_image_fraction": coverage,
                "reprojection_mean_px": float(np.mean(errors)) if errors.size else None,
                "reprojection_median_px": float(np.median(errors)) if errors.size else None,
                "reprojection_rms_px": rms,
                "reprojection_max_px": float(np.max(errors)) if errors.size else None,
                "neighbour_consistency_warning": neighbour_warning,
                "confidence": confidence,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if value is None:
                    formatted[key] = ""
                elif isinstance(value, float):
                    formatted[key] = f"{value:.9f}"
                elif isinstance(value, bool):
                    formatted[key] = str(value).lower()
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def _statistics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def build_summary(
    result: EstimationResult,
    rows: list[dict[str, Any]],
    nominal_step_deg: float,
    start_angle_deg: float,
) -> dict[str, Any]:
    expected_total = (len(rows) - 1) * nominal_step_deg
    steps = [row["step_deg"] for row in rows if row["step_deg"] is not None]
    step_errors = [value - nominal_step_deg for value in steps]
    step_stats = _statistics(steps)
    first_angle = rows[0]["estimated_angle_deg"]
    last_angle = rows[-1]["estimated_angle_deg"]
    measured_total = (
        last_angle - first_angle
        if first_angle is not None and last_angle is not None
        else None
    )
    closure = measured_total - expected_total if measured_total is not None else None
    used_observations = [item for item in result.observations if item.used]
    rejected_observations = [item for item in result.observations if not item.used]
    errors = [
        item.reprojection_errors
        for item in used_observations
        if item.reprojection_errors is not None
    ]
    all_errors = np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)
    table_normals = []
    for frame in result.frames:
        if frame.index in result.unresolved_frames:
            table_normals.append(None)
            continue
        world_normal = Rotation.from_rotvec(
            np.array([0.0, 0.0, result.state.theta[frame.index]])
        ).apply(result.world.reference_normal_world)
        table_normals.append(_vector(world_normal))

    return {
        "input": {
            "image_count": len(result.frames),
            "image_resolution": {
                "width": result.calibration.image_size[0],
                "height": result.calibration.image_size[1],
            },
            "marker_size_mm": result.marker_size_mm,
            "aruco_dictionary": result.dictionary_name,
            "intrinsics_file": str(result.calibration.path),
            "nominal_step_deg": nominal_step_deg,
            "start_angle_deg": start_angle_deg,
            "expected_total_rotation_deg": expected_total,
            "reference_image": (
                str(result.reference_frame.path) if result.reference_frame else None
            ),
        },
        "angle_results": {
            "mean_step_deg": step_stats["mean"],
            "median_step_deg": step_stats["median"],
            "step_standard_deviation_deg": step_stats["std"],
            "minimum_step_deg": step_stats["min"],
            "maximum_step_deg": step_stats["max"],
            "rms_step_error_deg": (
                float(np.sqrt(np.mean(np.square(step_errors))))
                if step_errors
                else None
            ),
            "maximum_absolute_step_error_deg": (
                float(np.max(np.abs(step_errors))) if step_errors else None
            ),
            "measured_total_rotation_deg": measured_total,
            "closure_error_deg": closure,
            "unresolved_frame_indices": sorted(result.unresolved_frames),
        },
        "geometry": {
            "rotation_axis_direction_world": _vector(result.world.axis_world),
            "point_on_rotation_axis_world_mm": _vector(
                result.world.axis_point_world
            ),
            "reference_table_normal_theta_0_world": _vector(
                result.world.reference_normal_world
            ),
            "table_normal_by_frame_world": table_normals,
            "axis_vs_reference_table_normal_deg": (
                result.world.axis_vs_reference_normal_deg
            ),
            "rotation_centre_world_mm": [0.0, 0.0, 0.0],
            "rotation_centre_camera_mm": _vector(result.world.origin_camera),
            "world_frame": {
                "origin": (
                    "intersection of the fitted rotation axis and the physical "
                    "reference table plane at theta=0"
                ),
                "z_axis": "fitted physical rotation axis in positive sequence direction",
                "x_axis": (
                    "camera-to-origin direction projected onto the plane perpendicular "
                    "to World Z and normalized"
                ),
                "y_axis": "World Z cross World X",
                "basis_vectors_in_opencv_camera_coordinates": {
                    "x": _vector(result.world.basis_camera[:, 0]),
                    "y": _vector(result.world.basis_camera[:, 1]),
                    "z": _vector(result.world.basis_camera[:, 2]),
                },
                "handedness": "right-handed",
                "note": (
                    "World XY is perpendicular to the rotation axis. The fitted "
                    "physical table plane is separate and rotates with theta."
                ),
            },
        },
        "camera": {
            "opencv_convention": "+X right, +Y down, +Z optical axis forward",
            "position_world_mm": _vector(result.world.camera_position_world),
            "optical_axis_world": _vector(result.world.optical_axis_world),
            "camera_centre_to_rotation_axis_distance_mm": (
                result.world.camera_to_axis_distance_mm
            ),
            "optical_axis_vs_rotation_axis_deg": (
                result.world.optical_axis_vs_rotation_axis_deg
            ),
            "downward_tilt_relative_world_xy_deg": (
                result.world.downward_tilt_world_xy_deg
            ),
            "downward_tilt_relative_reference_table_plane_theta_0_deg": (
                result.world.downward_tilt_reference_plane_deg
            ),
            "downward_tilt_sign_convention": (
                "plane normal is oriented toward the camera; horizontal is 0 deg "
                "and looking toward the plane is positive"
            ),
        },
        "optimization": {
            "unique_detected_marker_count": len(
                {item.marker_id for item in result.observations}
            ),
            "unique_detected_marker_ids": sorted(
                {item.marker_id for item in result.observations}
            ),
            "detected_observations": len(result.observations),
            "used_observations": len(used_observations),
            "rejected_observations": len(rejected_observations),
            "final_mean_reprojection_px": (
                float(np.mean(all_errors)) if all_errors.size else None
            ),
            "final_median_reprojection_px": (
                float(np.median(all_errors)) if all_errors.size else None
            ),
            "final_rms_reprojection_px": (
                float(np.sqrt(np.mean(all_errors * all_errors)))
                if all_errors.size
                else None
            ),
            "final_max_reprojection_px": (
                float(np.max(all_errors)) if all_errors.size else None
            ),
            "optimizer_success": result.diagnostics.success,
            "optimizer_status": result.diagnostics.status,
            "convergence_message": result.diagnostics.message,
            "cost": result.diagnostics.cost,
            "function_evaluations": result.diagnostics.evaluations,
            "optimality": result.diagnostics.optimality,
            "jacobian_rank": result.diagnostics.jacobian_rank,
            "jacobian_columns": result.diagnostics.jacobian_columns,
            "jacobian_condition": result.diagnostics.jacobian_condition,
            "outlier_threshold_px": result.diagnostics.outlier_threshold_px,
            "observation_graph_component_count": len(result.graph_components),
            "observation_graph": [
                {
                    "source_frames": [
                        value for value in component.frames if value < len(result.frames)
                    ],
                    "contains_reference_image": len(result.frames) in component.frames,
                    "marker_ids": list(component.marker_ids),
                }
                for component in result.graph_components
            ],
        },
        "confidence": {
            "rules": CONFIDENCE_RULES,
            "angle_uncertainty": {
                "kind": "approximate local 1-sigma",
                "assumptions": [
                    "camera intrinsics and marker size are exact",
                    "accepted corner errors are independent pixel noise",
                    "the rigid fixed-axis and planar-marker model is correct",
                    "the robust-loss Jacobian is a valid local linearization",
                ],
                "warning": (
                    "This is a fit-conditioning diagnostic, not guaranteed physical "
                    "accuracy. Calibration and marker-print errors are not included."
                ),
            },
        },
        "units": {
            "angles": "degrees",
            "lengths": "millimetres",
            "reprojection_error": "pixels",
        },
    }


def _save_line_plot(
    path: Path,
    x: np.ndarray,
    series: list[tuple[np.ndarray, str, dict[str, Any]]],
    title: str,
    ylabel: str,
    zero_line: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for values, label, style in series:
        axis.plot(x, values, label=label, **style)
    if zero_line is not None:
        axis.axhline(zero_line, color="black", linewidth=1.0, linestyle="--")
    axis.set_title(title)
    axis.set_xlabel("Frame index")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    if any(label for _, label, _ in series):
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    x = np.asarray([row["frame_index"] for row in rows], dtype=np.float64)

    def values(column: str) -> np.ndarray:
        return np.asarray(
            [np.nan if row[column] is None else row[column] for row in rows],
            dtype=np.float64,
        )

    paths: list[Path] = []
    plot_specs = [
        (
            "measured_angle_vs_nominal.png",
            [
                (values("estimated_angle_deg"), "measured", {"linewidth": 2}),
                (
                    values("nominal_angle_deg"),
                    "nominal",
                    {"linewidth": 1.5, "linestyle": "--"},
                ),
            ],
            "Measured physical angle vs nominal angle",
            "Angle (deg)",
            None,
        ),
        (
            "step_angle.png",
            [
                (values("step_deg"), "measured step", {"marker": "."}),
                (
                    np.full_like(x, rows[1]["nominal_angle_deg"] - rows[0]["nominal_angle_deg"]),
                    "nominal step",
                    {"linestyle": "--"},
                ),
            ],
            "Measured angular step",
            "Step (deg)",
            None,
        ),
        (
            "step_error.png",
            [(values("step_error_deg"), "step error", {"marker": "."})],
            "Step error: measured minus nominal",
            "Error (deg)",
            0.0,
        ),
        (
            "cumulative_angle_error.png",
            [
                (
                    values("cumulative_error_deg"),
                    "cumulative error",
                    {"linewidth": 2},
                )
            ],
            "Cumulative measured-angle error",
            "Error (deg)",
            0.0,
        ),
        (
            "reprojection_error.png",
            [
                (
                    values("reprojection_rms_px"),
                    "RMS reprojection error",
                    {"marker": "."},
                )
            ],
            "Per-frame reprojection error",
            "RMS error (px)",
            0.0,
        ),
        (
            "angle_uncertainty.png",
            [
                (
                    values("angle_uncertainty_deg"),
                    "approximate 1-sigma",
                    {"marker": "."},
                )
            ],
            "Approximate angle uncertainty",
            "Uncertainty (deg)",
            0.0,
        ),
    ]
    for filename, series, title, ylabel, zero_line in plot_specs:
        path = output_dir / filename
        _save_line_plot(path, x, series, title, ylabel, zero_line)
        paths.append(path)

    path = output_dir / "marker_visibility.png"
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.plot(x, values("detected_marker_count"), label="detected markers", marker=".")
    axis.plot(x, values("used_marker_count"), label="used markers", marker=".")
    axis.set_xlabel("Frame index")
    axis.set_ylabel("Marker count")
    axis.grid(True, alpha=0.3)
    second = axis.twinx()
    second.plot(
        x,
        values("used_corner_count"),
        label="used corners",
        color="tab:green",
        alpha=0.7,
    )
    second.set_ylabel("Corner count")
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, loc="best")
    axis.set_title("Marker visibility and accepted corner support")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    return paths


def write_top_view(output_dir: Path, result: EstimationResult) -> Path:
    path = output_dir / "turntable_top_view.png"
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    for marker_id, corners in sorted(result.world.marker_corners_world.items()):
        closed = np.vstack((corners[:, :2], corners[0, :2]))
        axis.plot(closed[:, 0], closed[:, 1], linewidth=1.5)
        center = result.world.marker_centers_world[marker_id]
        axis.text(center[0], center[1], str(marker_id), ha="center", va="center")
    axis.scatter([0.0], [0.0], marker="x", s=80, color="black", label="rotation centre")
    extents = [
        np.linalg.norm(center[:2])
        for center in result.world.marker_centers_world.values()
    ]
    scale = max(extents + [result.marker_size_mm]) * 0.35
    axis.arrow(0, 0, scale, 0, width=scale * 0.01, color="red", length_includes_head=True)
    axis.arrow(0, 0, 0, scale, width=scale * 0.01, color="green", length_includes_head=True)
    axis.text(scale, 0, " X", color="red")
    axis.text(0, scale, " Y", color="green")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("World X (mm)")
    axis.set_ylabel("World Y (mm)")
    axis.set_title("Recovered marker layout, viewed along the rotation axis")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_geometry_overview(output_dir: Path, result: EstimationResult) -> Path:
    path = output_dir / "geometry_overview.png"
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    all_corners = np.vstack(list(result.world.marker_corners_world.values()))
    extent = max(float(np.ptp(all_corners[:, 0])), float(np.ptp(all_corners[:, 1])), result.marker_size_mm)
    for marker_id, corners in sorted(result.world.marker_corners_world.items()):
        closed = np.vstack((corners, corners[0]))
        axis.plot(closed[:, 0], closed[:, 1], closed[:, 2])
        center = result.world.marker_centers_world[marker_id]
        axis.text(*center, str(marker_id))

    normal = result.world.reference_normal_world
    grid = np.linspace(-0.6 * extent, 0.6 * extent, 2)
    xx, yy = np.meshgrid(grid, grid)
    if abs(normal[2]) > 1e-8:
        zz = -(normal[0] * xx + normal[1] * yy) / normal[2]
        axis.plot_surface(xx, yy, zz, alpha=0.12, color="tab:blue")
    axis.plot([0, 0], [0, 0], [-extent, extent], color="black", linewidth=2, label="rotation axis")
    colors = ("red", "green", "blue")
    for dimension, (label, color) in enumerate(zip(("X", "Y", "Z"), colors)):
        endpoint = np.zeros(3)
        endpoint[dimension] = 0.35 * extent
        axis.plot([0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]], color=color)
        axis.text(*endpoint, label, color=color)
    camera = result.world.camera_position_world
    optical_end = camera + result.world.optical_axis_world * (0.4 * extent)
    axis.scatter(*camera, color="magenta", marker="^", s=60, label="camera centre")
    axis.plot(
        [camera[0], optical_end[0]],
        [camera[1], optical_end[1]],
        [camera[2], optical_end[2]],
        color="magenta",
        label="camera optical axis",
    )
    axis.scatter(0, 0, 0, color="black", marker="x", s=70)
    axis.set_xlabel("World X (mm)")
    axis.set_ylabel("World Y (mm)")
    axis.set_zlabel("World Z (mm)")
    axis.set_title("Recovered turntable and camera geometry")
    axis.legend(loc="best")
    axis.set_box_aspect((1, 1, 1))
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _draw_label(image: np.ndarray, lines: list[str]) -> None:
    height, width = image.shape[:2]
    scale = max(0.55, min(1.2, width / 2200.0))
    thickness = max(1, round(scale * 2))
    line_height = round(27 * scale)
    margin = round(12 * scale)
    box_width = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        for line in lines
    ) + 2 * margin
    box_height = len(lines) * line_height + 2 * margin
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (margin, margin + (index + 1) * line_height - round(5 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def write_annotated_frames(
    output_dir: Path,
    result: EstimationResult,
    rows: list[dict[str, Any]],
) -> Path:
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    origin = result.world.origin_camera
    basis = result.world.basis_camera
    marker_extent = max(
        [
            np.linalg.norm(center - origin)
            for center in result.state.marker_centers.values()
        ]
        + [result.marker_size_mm]
    )
    overlay_scale = max(result.marker_size_mm, 0.25 * marker_extent)
    geometry_points = np.vstack(
        (
            origin,
            origin + basis[:, 0] * overlay_scale,
            origin + basis[:, 1] * overlay_scale,
            origin + basis[:, 2] * overlay_scale,
            origin - basis[:, 2] * overlay_scale,
        )
    )
    projected = project_camera_points(geometry_points, result.calibration)
    centre = tuple(np.rint(projected[0]).astype(int))
    for frame, row in zip(result.frames, rows):
        image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"could not reread image for annotation: {frame.path}")
        for observation in frame.observations:
            color = (40, 200, 40) if observation.used else (30, 30, 230)
            corners = np.rint(observation.corners).astype(np.int32)
            cv2.polylines(image, [corners], True, color, 3, cv2.LINE_AA)
            cv2.putText(
                image,
                str(observation.marker_id),
                tuple(corners[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.drawMarker(image, centre, (0, 255, 255), cv2.MARKER_CROSS, 24, 3)
        cv2.line(image, tuple(np.rint(projected[4]).astype(int)), tuple(np.rint(projected[3]).astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
        for point, color, label in zip(
            projected[1:4], ((0, 0, 255), (0, 200, 0), (255, 0, 0)), ("X", "Y", "Z")
        ):
            endpoint = tuple(np.rint(point).astype(int))
            cv2.arrowedLine(image, centre, endpoint, color, 3, cv2.LINE_AA, tipLength=0.12)
            cv2.putText(image, label, endpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        def display(value: Any, suffix: str = "") -> str:
            return "unresolved" if value is None else f"{value:.4f}{suffix}"

        lines = [
            f"frame {frame.index}: {frame.filename}",
            f"nominal: {row['nominal_angle_deg']:.4f} deg",
            f"measured: {display(row['estimated_angle_deg'], ' deg')}",
            f"step: {display(row['step_deg'], ' deg')}  error: {display(row['step_error_deg'], ' deg')}",
            f"markers: {row['used_marker_count']}/{row['detected_marker_count']}  corners: {row['used_corner_count']}",
            f"reprojection RMS: {display(row['reprojection_rms_px'], ' px')}",
            f"confidence: {row['confidence']}",
        ]
        _draw_label(image, lines)
        destination = annotated_dir / f"{frame.index:04d}_{frame.path.stem}.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"could not write annotated image: {destination}")
    return annotated_dir


def write_all_outputs(
    output_dir: Path,
    result: EstimationResult,
    nominal_step_deg: float,
    start_angle_deg: float,
    annotated_images: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_frame_rows(result, nominal_step_deg, start_angle_deg)
    summary = build_summary(result, rows, nominal_step_deg, start_angle_deg)
    csv_path = output_dir / "turntable_angles.csv"
    summary_path = output_dir / "summary.json"
    _write_csv(csv_path, rows)
    with summary_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    plots = write_plots(output_dir, rows)
    plots.append(write_top_view(output_dir, result))
    plots.append(write_geometry_overview(output_dir, result))
    annotated_dir = (
        write_annotated_frames(output_dir, result, rows) if annotated_images else None
    )
    outputs = {
        "csv": csv_path,
        "summary": summary_path,
        "plots": plots,
        "annotated": annotated_dir,
    }
    return rows, summary, outputs
