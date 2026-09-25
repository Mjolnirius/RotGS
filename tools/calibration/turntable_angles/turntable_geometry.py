"""Geometry and optimization for physical turntable-angle estimation.

The camera coordinate system is the optimization coordinate system: OpenCV +X
points right, +Y points down, and +Z points forward.  A marker layout is defined
in the physical table plane at theta=0 and is rotated rigidly about one fixed 3D
axis for every other frame.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_NATURAL_PART = re.compile(r"(\d+)")


class EstimationError(RuntimeError):
    """Raised when the observations cannot support a trustworthy estimate."""


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    path: Path


@dataclass
class PoseCandidate:
    rotation: np.ndarray
    translation: np.ndarray
    reprojection_rms: float

    @property
    def transform(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.rotation
        result[:3, 3] = self.translation
        return result


@dataclass
class MarkerObservation:
    frame_key: int
    marker_id: int
    corners: np.ndarray
    perimeter_px: float
    is_reference: bool = False
    candidates: list[PoseCandidate] = field(default_factory=list)
    selected_candidate: int = 0
    used: bool = True
    reprojection_errors: np.ndarray | None = None
    rejection_reason: str | None = None


@dataclass
class FrameRecord:
    index: int
    path: Path
    filename: str
    observations: list[MarkerObservation]
    image_size: tuple[int, int]
    rejected_detection_count: int = 0
    is_reference: bool = False


@dataclass(frozen=True)
class GraphComponent:
    frames: tuple[int, ...]
    marker_ids: tuple[int, ...]


@dataclass
class DecodedState:
    axis: np.ndarray
    axis_point: np.ndarray
    plane_normal: np.ndarray
    plane_offset: float
    marker_centers: dict[int, np.ndarray]
    marker_x_axes: dict[int, np.ndarray]
    marker_y_axes: dict[int, np.ndarray]
    marker_corners: dict[int, np.ndarray]
    theta: dict[int, float]


@dataclass
class OptimizationDiagnostics:
    success: bool
    status: int
    message: str
    cost: float
    evaluations: int
    optimality: float
    jacobian_rank: int
    jacobian_columns: int
    jacobian_condition: float | None
    outlier_threshold_px: float


@dataclass
class WorldGeometry:
    origin_camera: np.ndarray
    basis_camera: np.ndarray
    axis_world: np.ndarray
    axis_point_world: np.ndarray
    reference_normal_world: np.ndarray
    marker_centers_world: dict[int, np.ndarray]
    marker_corners_world: dict[int, np.ndarray]
    camera_position_world: np.ndarray
    optical_axis_world: np.ndarray
    camera_to_axis_distance_mm: float
    optical_axis_vs_rotation_axis_deg: float
    downward_tilt_world_xy_deg: float
    downward_tilt_reference_plane_deg: float
    axis_vs_reference_normal_deg: float


@dataclass
class EstimationResult:
    frames: list[FrameRecord]
    reference_frame: FrameRecord | None
    calibration: CameraCalibration
    marker_size_mm: float
    dictionary_name: str
    state: DecodedState
    world: WorldGeometry
    diagnostics: OptimizationDiagnostics
    graph_components: list[GraphComponent]
    angle_uncertainty_deg: dict[int, float | None]
    unresolved_frames: set[int]

    @property
    def observations(self) -> list[MarkerObservation]:
        items = [obs for frame in self.frames for obs in frame.observations]
        if self.reference_frame is not None:
            items.extend(self.reference_frame.observations)
        return items


def natural_sort_key(path: Path) -> tuple[tuple[int, object], ...]:
    """Return a deterministic, case-insensitive natural filename key."""
    parts: list[tuple[int, object]] = []
    for part in _NATURAL_PART.split(path.name.casefold()):
        if part.isdigit():
            parts.append((1, int(part)))
        else:
            parts.append((0, part))
    parts.append((0, path.name))
    return tuple(parts)


def discover_images(folder: Path) -> list[Path]:
    folder = folder.expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError(f"image folder is not a directory: {folder}")
    images = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
    ]
    images.sort(key=natural_sort_key)
    if not images:
        raise ValueError(f"no JPG, JPEG, or PNG images found in: {folder}")
    return images


def load_calibration(path: Path) -> CameraCalibration:
    calibration_path = path.expanduser().resolve(strict=True)
    if not calibration_path.is_file():
        raise ValueError(f"intrinsics path is not a file: {calibration_path}")
    try:
        with np.load(calibration_path, allow_pickle=False) as calibration:
            required = {"camera_matrix", "dist_coeffs", "image_size"}
            missing = required.difference(calibration.files)
            if missing:
                raise ValueError(
                    "calibration file is missing fields: "
                    + ", ".join(sorted(missing))
                )
            camera_matrix = np.asarray(
                calibration["camera_matrix"], dtype=np.float64
            )
            distortion = np.asarray(
                calibration["dist_coeffs"], dtype=np.float64
            ).reshape(-1)
            raw_size = np.asarray(calibration["image_size"]).reshape(-1)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and "calibration file" in str(exc):
            raise
        raise ValueError(
            f"could not read OpenCV NPZ calibration: {calibration_path}"
        ) from exc

    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise ValueError("calibration camera_matrix must be a finite 3x3 matrix")
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise ValueError("calibration focal lengths must be positive")
    if not np.allclose(camera_matrix[2], (0.0, 0.0, 1.0), atol=1e-12):
        raise ValueError("calibration camera_matrix must end with [0, 0, 1]")
    if distortion.size not in {4, 5, 8, 12, 14} or not np.all(
        np.isfinite(distortion)
    ):
        raise ValueError(
            "calibration dist_coeffs must contain 4, 5, 8, 12, or 14 "
            "finite values"
        )
    if raw_size.size != 2:
        raise ValueError("calibration image_size must contain width and height")
    image_size = (int(raw_size[0]), int(raw_size[1]))
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("calibration image_size values must be positive")
    return CameraCalibration(camera_matrix, distortion, image_size, calibration_path)


def marker_object_corners(marker_size_mm: float) -> np.ndarray:
    half = marker_size_mm / 2.0
    # Required ordering for OpenCV SOLVEPNP_IPPE_SQUARE.
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def resolve_aruco_dictionary(name: str):
    if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        names = sorted(
            item
            for item in dir(cv2.aruco)
            if item.startswith("DICT_") and isinstance(getattr(cv2.aruco, item), int)
        )
        raise ValueError(
            f"unknown ArUco dictionary {name!r}; available values include: "
            + ", ".join(names)
        )
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _pose_candidates(
    corners: np.ndarray,
    calibration: CameraCalibration,
    marker_size_mm: float,
) -> list[PoseCandidate]:
    object_points = marker_object_corners(marker_size_mm)
    result = cv2.solvePnPGeneric(
        object_points,
        np.asarray(corners, dtype=np.float64),
        calibration.camera_matrix,
        calibration.distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not result or not result[0]:
        return []
    rvecs, tvecs = result[1], result[2]
    candidates: list[PoseCandidate] = []
    for rvec, tvec in zip(rvecs, tvecs):
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        camera_points = object_points @ rotation.T + translation
        if np.min(camera_points[:, 2]) <= 1e-6:
            continue
        projected, _ = cv2.projectPoints(
            object_points,
            np.asarray(rvec, dtype=np.float64),
            translation,
            calibration.camera_matrix,
            calibration.distortion,
        )
        delta = projected.reshape(4, 2) - corners
        rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        candidates.append(PoseCandidate(rotation, translation, rms))
    candidates.sort(key=lambda item: item.reprojection_rms)
    return candidates


def detect_frames(
    image_paths: Sequence[Path],
    calibration: CameraCalibration,
    marker_size_mm: float,
    dictionary_name: str,
    reference_image: Path | None = None,
    verbose: bool = False,
) -> tuple[list[FrameRecord], FrameRecord | None]:
    """Detect and subpixel-refine ArUco corners in source and reference images."""
    dictionary = resolve_aruco_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect_one(path: Path, frame_key: int, is_reference: bool) -> FrameRecord:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {path}")
        size = (int(image.shape[1]), int(image.shape[0]))
        if size != calibration.image_size:
            raise ValueError(
                "calibration and image dimensions differ; intrinsics scaling is "
                "not implicit: "
                f"calibration={calibration.image_size[0]}x{calibration.image_size[1]}, "
                f"image={size[0]}x{size[1]} ({path.name})"
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners_list, ids, rejected = detector.detectMarkers(gray)
        observations: list[MarkerObservation] = []
        duplicate_rejections = 0
        if ids is not None:
            grouped: dict[int, list[np.ndarray]] = defaultdict(list)
            for corners, marker_id in zip(corners_list, ids.reshape(-1)):
                refined32 = np.asarray(corners, dtype=np.float32).reshape(4, 2).copy()
                cv2.cornerSubPix(
                    gray,
                    refined32,
                    (5, 5),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                        50,
                        0.01,
                    ),
                )
                refined = refined32.astype(np.float64)
                grouped[int(marker_id)].append(refined)
            for marker_id, detections in grouped.items():
                if len(detections) > 1:
                    detections.sort(
                        key=lambda value: -float(
                            cv2.arcLength(value.astype(np.float32), True)
                        )
                    )
                    duplicate_rejections += len(detections) - 1
                corners = detections[0]
                perimeter = float(cv2.arcLength(corners.astype(np.float32), True))
                candidates = _pose_candidates(corners, calibration, marker_size_mm)
                if not candidates:
                    duplicate_rejections += 1
                    continue
                observations.append(
                    MarkerObservation(
                        frame_key=frame_key,
                        marker_id=marker_id,
                        corners=corners,
                        perimeter_px=perimeter,
                        is_reference=is_reference,
                        candidates=candidates,
                    )
                )
        observations.sort(key=lambda item: item.marker_id)
        if verbose:
            label = "reference" if is_reference else f"frame {frame_key}"
            print(
                f"Detection {label:>12}: {path.name} — "
                f"{len(observations)} markers, {len(rejected)} rejected candidates"
            )
        return FrameRecord(
            index=frame_key,
            path=path,
            filename=path.name,
            observations=observations,
            image_size=size,
            rejected_detection_count=len(rejected) + duplicate_rejections,
            is_reference=is_reference,
        )

    frames = [detect_one(path, index, False) for index, path in enumerate(image_paths)]
    reference: FrameRecord | None = None
    if reference_image is not None:
        reference_path = reference_image.expanduser().resolve(strict=True)
        matching = [frame for frame in frames if frame.path.resolve() == reference_path]
        if matching:
            if verbose:
                print(
                    "Reference image is already part of the source sequence; "
                    "reusing its observation instead of duplicating it."
                )
        else:
            reference = detect_one(reference_path, len(frames), True)
    return frames, reference


def observation_graph_components(
    observations: Iterable[MarkerObservation],
    used_only: bool = False,
) -> list[GraphComponent]:
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for observation in observations:
        if used_only and not observation.used:
            continue
        frame_node = ("frame", observation.frame_key)
        marker_node = ("marker", observation.marker_id)
        adjacency[frame_node].add(marker_node)
        adjacency[marker_node].add(frame_node)
    components: list[GraphComponent] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        queue = deque([start])
        visited: set[tuple[str, int]] = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adjacency[node] - visited)
        remaining.difference_update(visited)
        components.append(
            GraphComponent(
                tuple(sorted(value for kind, value in visited if kind == "frame")),
                tuple(sorted(value for kind, value in visited if kind == "marker")),
            )
        )
    components.sort(key=lambda item: item.frames)
    return components


def _component_error(components: Sequence[GraphComponent], frame_count: int) -> str:
    lines = ["Marker observation graph is disconnected."]
    for index, component in enumerate(components, start=1):
        source_frames = [value for value in component.frames if value < frame_count]
        reference = frame_count in component.frames
        lines.append(
            f"  component {index}: frames={source_frames}, "
            f"reference_image={reference}, markers={list(component.marker_ids)}"
        )
    lines.append(
        "Add marker overlap between these frame groups or provide an empty-turntable "
        "reference image that sees markers from every group."
    )
    return "\n".join(lines)


def validate_observation_graph(
    observations: Sequence[MarkerObservation],
    frame_count: int,
    used_only: bool = False,
) -> list[GraphComponent]:
    selected = [item for item in observations if item.used or not used_only]
    source_with_observations = {
        item.frame_key for item in selected if item.frame_key < frame_count
    }
    if 0 not in source_with_observations:
        raise EstimationError(
            "Frame 0 has no usable marker observations, so theta_0 cannot anchor "
            "the physical sequence."
        )
    unique_markers = {item.marker_id for item in selected}
    if len(unique_markers) < 2:
        raise EstimationError(
            "At least two unique ArUco markers are required for a robust rigid "
            "layout and fixed-axis estimate."
        )
    if len(source_with_observations) < 3:
        raise EstimationError(
            "At least three source frames with usable marker observations are required."
        )
    components = observation_graph_components(selected, used_only=False)
    if len(components) != 1:
        raise EstimationError(_component_error(components, frame_count))
    return components


def _inverse_transform(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(Rotation.from_matrix(left.T @ right).magnitude())


def _pose_distance(
    left: np.ndarray, right: np.ndarray, marker_size_mm: float
) -> float:
    angular = _rotation_distance(left[:3, :3], right[:3, :3])
    translation = np.linalg.norm(left[:3, 3] - right[:3, 3]) / marker_size_mm
    return angular + 0.35 * float(translation)


def _average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("cannot average an empty transform sequence")
    rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item in transforms]))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotations.mean().as_matrix()
    result[:3, 3] = np.median(
        np.stack([item[:3, 3] for item in transforms]), axis=0
    )
    return result


def _selected_transform(observation: MarkerObservation) -> np.ndarray:
    return observation.candidates[observation.selected_candidate].transform


def _fit_plane(points: np.ndarray, preferred_normal: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if float(np.dot(normal, preferred_normal)) < 0.0:
        normal = -normal
    return normal, float(np.dot(normal, centroid))


def _select_reference_branches(
    observations: Sequence[MarkerObservation], marker_size_mm: float
) -> None:
    """Choose frame-zero IPPE branches with a shared-plane consensus."""
    if not observations:
        return
    best_score = math.inf
    best_selection: list[int] | None = None
    for seed_observation in observations:
        for seed_candidate in seed_observation.candidates:
            normal = seed_candidate.rotation[:, 2]
            offset = float(np.dot(normal, seed_candidate.translation))
            selection: list[int] = []
            score = 0.0
            for observation in observations:
                candidate_scores = []
                for candidate in observation.candidates:
                    alignment = float(
                        np.arccos(
                            np.clip(
                                abs(np.dot(normal, candidate.rotation[:, 2])),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    plane_distance = abs(
                        np.dot(normal, candidate.translation) - offset
                    ) / marker_size_mm
                    candidate_scores.append(
                        4.0 * alignment
                        + plane_distance
                        + 0.1 * candidate.reprojection_rms
                    )
                selected = int(np.argmin(candidate_scores))
                selection.append(selected)
                score += candidate_scores[selected]
            if score < best_score:
                best_score = score
                best_selection = selection
    assert best_selection is not None
    for observation, selected in zip(observations, best_selection):
        observation.selected_candidate = selected


def _initial_pose_factorization(
    observations: Sequence[MarkerObservation], marker_size_mm: float
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    by_frame: dict[int, list[MarkerObservation]] = defaultdict(list)
    for observation in observations:
        by_frame[observation.frame_key].append(observation)
    _select_reference_branches(by_frame[0], marker_size_mm)

    frame_transforms: dict[int, np.ndarray] = {0: np.eye(4, dtype=np.float64)}
    marker_transforms: dict[int, np.ndarray] = {
        observation.marker_id: _selected_transform(observation)
        for observation in by_frame[0]
    }

    while True:
        progress = False
        for frame_key, frame_observations in sorted(by_frame.items()):
            if frame_key in frame_transforms:
                frame_transform = frame_transforms[frame_key]
                for observation in frame_observations:
                    if observation.marker_id in marker_transforms:
                        continue
                    candidates = [
                        _inverse_transform(frame_transform) @ candidate.transform
                        for candidate in observation.candidates
                    ]
                    if marker_transforms:
                        existing_normals = np.stack(
                            [value[:3, 2] for value in marker_transforms.values()]
                        )
                        preferred = np.mean(existing_normals, axis=0)
                        preferred /= np.linalg.norm(preferred)
                        centers = np.stack(
                            [value[:3, 3] for value in marker_transforms.values()]
                        )
                        offset = float(np.median(centers @ preferred))
                        scores = [
                            4.0
                            * np.arccos(
                                np.clip(abs(np.dot(preferred, value[:3, 2])), -1, 1)
                            )
                            + abs(np.dot(preferred, value[:3, 3]) - offset)
                            / marker_size_mm
                            + 0.1 * observation.candidates[index].reprojection_rms
                            for index, value in enumerate(candidates)
                        ]
                        observation.selected_candidate = int(np.argmin(scores))
                    else:
                        observation.selected_candidate = 0
                    marker_transforms[observation.marker_id] = candidates[
                        observation.selected_candidate
                    ]
                    progress = True

        for frame_key, frame_observations in sorted(by_frame.items()):
            if frame_key in frame_transforms:
                continue
            proposals: list[np.ndarray] = []
            for observation in frame_observations:
                marker_transform = marker_transforms.get(observation.marker_id)
                if marker_transform is None:
                    continue
                inverse_marker = _inverse_transform(marker_transform)
                proposals.extend(
                    candidate.transform @ inverse_marker
                    for candidate in observation.candidates
                )
            if not proposals:
                continue
            best_proposal: np.ndarray | None = None
            best_score = math.inf
            best_indices: dict[int, int] = {}
            for proposal in proposals:
                score = 0.0
                indices: dict[int, int] = {}
                for observation in frame_observations:
                    marker_transform = marker_transforms.get(observation.marker_id)
                    if marker_transform is None:
                        continue
                    predicted = proposal @ marker_transform
                    candidate_scores = [
                        _pose_distance(
                            predicted, candidate.transform, marker_size_mm
                        )
                        + 0.05 * candidate.reprojection_rms
                        for candidate in observation.candidates
                    ]
                    selected = int(np.argmin(candidate_scores))
                    indices[observation.marker_id] = selected
                    score += min(candidate_scores)
                if score < best_score:
                    best_score = score
                    best_proposal = proposal
                    best_indices = indices
            assert best_proposal is not None
            chosen_transforms = []
            for observation in frame_observations:
                if observation.marker_id not in best_indices:
                    continue
                observation.selected_candidate = best_indices[observation.marker_id]
                chosen_transforms.append(
                    _selected_transform(observation)
                    @ _inverse_transform(marker_transforms[observation.marker_id])
                )
            frame_transforms[frame_key] = _average_transforms(chosen_transforms)
            progress = True
        if not progress:
            break

    if set(by_frame) != set(frame_transforms):
        missing = sorted(set(by_frame) - set(frame_transforms))
        raise EstimationError(
            "could not propagate a consistent marker layout to frames: "
            + ", ".join(map(str, missing))
        )

    # Alternate pose/candidate synchronization to reduce spanning-tree drift.
    for _ in range(6):
        marker_updates: dict[int, list[np.ndarray]] = defaultdict(list)
        for observation in observations:
            predicted = (
                frame_transforms[observation.frame_key]
                @ marker_transforms[observation.marker_id]
            )
            scores = [
                _pose_distance(predicted, candidate.transform, marker_size_mm)
                + 0.05 * candidate.reprojection_rms
                for candidate in observation.candidates
            ]
            observation.selected_candidate = int(np.argmin(scores))
            marker_updates[observation.marker_id].append(
                _inverse_transform(frame_transforms[observation.frame_key])
                @ _selected_transform(observation)
            )
        marker_transforms = {
            marker_id: _average_transforms(values)
            for marker_id, values in marker_updates.items()
        }
        frame_updates: dict[int, list[np.ndarray]] = defaultdict(list)
        for observation in observations:
            frame_updates[observation.frame_key].append(
                _selected_transform(observation)
                @ _inverse_transform(marker_transforms[observation.marker_id])
            )
        for frame_key, values in frame_updates.items():
            if frame_key != 0:
                frame_transforms[frame_key] = _average_transforms(values)
        frame_transforms[0] = np.eye(4, dtype=np.float64)
    return frame_transforms, marker_transforms


def _signed_rotation_angle(rotation: np.ndarray, axis: np.ndarray) -> float:
    sine = 0.5 * float(
        np.dot(
            axis,
            np.array(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ]
            ),
        )
    )
    cosine = 0.5 * (float(np.trace(rotation)) - 1.0)
    return math.atan2(sine, cosine)


def _unwrap_near(value: float, target: float) -> float:
    return value + 2.0 * math.pi * round((target - value) / (2.0 * math.pi))


def _initial_axis_and_angles(
    frame_transforms: dict[int, np.ndarray],
    source_frame_count: int,
    nominal_step_rad: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    matrices = []
    for frame_key, transform in frame_transforms.items():
        if frame_key == 0:
            continue
        rotation = transform[:3, :3]
        if Rotation.from_matrix(rotation).magnitude() > math.radians(1.0):
            matrices.append((rotation - np.eye(3)).T @ (rotation - np.eye(3)))
    if len(matrices) < 2:
        raise EstimationError(
            "The observed sequence has insufficient angular span to determine a "
            "stable rotation axis."
        )
    _, eigenvectors = np.linalg.eigh(np.sum(matrices, axis=0))
    unsigned_axis = eigenvectors[:, 0]
    unsigned_axis /= np.linalg.norm(unsigned_axis)

    def angles_and_cost(axis: np.ndarray) -> tuple[dict[int, float], float]:
        angles = {0: 0.0}
        cost = 0.0
        for frame_key, transform in frame_transforms.items():
            if frame_key == 0:
                continue
            raw = _signed_rotation_angle(transform[:3, :3], axis)
            if frame_key < source_frame_count:
                target = frame_key * nominal_step_rad
                angle = _unwrap_near(raw, target)
                cost += (angle - target) ** 2
            else:
                angle = raw
            angles[frame_key] = angle
        return angles, cost

    positive_angles, positive_cost = angles_and_cost(unsigned_axis)
    negative_angles, negative_cost = angles_and_cost(-unsigned_axis)
    if negative_cost < positive_cost:
        axis, angles = -unsigned_axis, negative_angles
    else:
        axis, angles = unsigned_axis, positive_angles

    lhs: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    for frame_key, transform in frame_transforms.items():
        if frame_key == 0:
            continue
        rotation = Rotation.from_rotvec(axis * angles[frame_key]).as_matrix()
        lhs.append(np.eye(3) - rotation)
        rhs.append(transform[:3, 3])
    lhs.append(axis.reshape(1, 3))
    rhs.append(np.zeros(1, dtype=np.float64))
    axis_point, *_ = np.linalg.lstsq(np.vstack(lhs), np.concatenate(rhs), rcond=None)
    axis_point -= axis * float(np.dot(axis, axis_point))
    return axis, axis_point, angles


def _initial_plane_and_layout(
    marker_transforms: dict[int, np.ndarray], marker_size_mm: float
) -> tuple[np.ndarray, float, dict[int, tuple[float, float, float]]]:
    object_points = marker_object_corners(marker_size_mm)
    all_points = np.vstack(
        [_transform_points(transform, object_points) for transform in marker_transforms.values()]
    )
    preferred = np.mean(
        np.stack([transform[:3, 2] for transform in marker_transforms.values()]), axis=0
    )
    preferred /= np.linalg.norm(preferred)
    normal, offset = _fit_plane(all_points, preferred)
    seed = int(np.argmin(np.abs(normal)))
    plane_x, plane_y = _stable_basis(normal, seed)
    origin = offset * normal
    marker_values: dict[int, tuple[float, float, float]] = {}
    for marker_id, transform in marker_transforms.items():
        center = transform[:3, 3]
        center -= normal * (float(np.dot(normal, center)) - offset)
        local = center - origin
        marker_x = transform[:3, 0]
        marker_x -= normal * float(np.dot(normal, marker_x))
        marker_x /= np.linalg.norm(marker_x)
        marker_values[marker_id] = (
            float(np.dot(local, plane_x)),
            float(np.dot(local, plane_y)),
            math.atan2(float(np.dot(marker_x, plane_y)), float(np.dot(marker_x, plane_x))),
        )
    return normal, offset, marker_values


def _stable_basis(direction: np.ndarray, seed_index: int) -> tuple[np.ndarray, np.ndarray]:
    seed = np.eye(3, dtype=np.float64)[seed_index]
    first = seed - direction * float(np.dot(seed, direction))
    norm = np.linalg.norm(first)
    if norm < 1e-8:
        raise EstimationError("direction parameterization became numerically singular")
    first /= norm
    second = np.cross(direction, first)
    second /= np.linalg.norm(second)
    return first, second


class _BundleModel:
    def __init__(
        self,
        observations: Sequence[MarkerObservation],
        calibration: CameraCalibration,
        marker_size_mm: float,
        axis: np.ndarray,
        axis_point: np.ndarray,
        plane_normal: np.ndarray,
        plane_offset: float,
        marker_values: dict[int, tuple[float, float, float]],
        theta: dict[int, float],
        source_frame_count: int,
    ) -> None:
        self.observations = list(observations)
        self.calibration = calibration
        self.marker_size_mm = marker_size_mm
        self.object_corners = marker_object_corners(marker_size_mm)
        self.axis_anchor = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
        self.plane_anchor = np.asarray(plane_normal, dtype=np.float64) / np.linalg.norm(
            plane_normal
        )
        self.axis_seed = int(np.argmin(np.abs(self.axis_anchor)))
        self.plane_seed = int(np.argmin(np.abs(self.plane_anchor)))
        self.axis_tangent = _stable_basis(self.axis_anchor, self.axis_seed)
        self.plane_tangent = _stable_basis(self.plane_anchor, self.plane_seed)
        self.marker_ids = sorted({item.marker_id for item in observations})
        self.frame_keys = sorted({item.frame_key for item in observations})
        if 0 not in self.frame_keys:
            raise EstimationError("frame 0 is absent from the bundle-adjustment graph")
        self.source_frame_count = source_frame_count
        self.marker_slices: dict[int, slice] = {}
        self.theta_indices: dict[int, int] = {}

        values = [0.0, 0.0]
        axis_u, axis_v = _stable_basis(self.axis_anchor, self.axis_seed)
        values.extend(
            [float(np.dot(axis_point, axis_u)), float(np.dot(axis_point, axis_v))]
        )
        values.extend([0.0, 0.0, float(plane_offset)])
        for marker_id in self.marker_ids:
            start = len(values)
            values.extend(marker_values[marker_id])
            self.marker_slices[marker_id] = slice(start, start + 3)
        for frame_key in self.frame_keys:
            if frame_key == 0:
                continue
            self.theta_indices[frame_key] = len(values)
            values.append(theta[frame_key])
        self.initial = np.asarray(values, dtype=np.float64)

    def _direction(
        self, anchor: np.ndarray, tangent: tuple[np.ndarray, np.ndarray], delta: np.ndarray
    ) -> np.ndarray:
        direction = anchor + delta[0] * tangent[0] + delta[1] * tangent[1]
        return direction / np.linalg.norm(direction)

    def decode(self, parameters: np.ndarray) -> DecodedState:
        axis = self._direction(self.axis_anchor, self.axis_tangent, parameters[0:2])
        axis_u, axis_v = _stable_basis(axis, self.axis_seed)
        axis_point = parameters[2] * axis_u + parameters[3] * axis_v
        plane_normal = self._direction(
            self.plane_anchor, self.plane_tangent, parameters[4:6]
        )
        plane_offset = float(parameters[6])
        plane_x, plane_y = _stable_basis(plane_normal, self.plane_seed)
        plane_origin = plane_offset * plane_normal
        marker_centers: dict[int, np.ndarray] = {}
        marker_x_axes: dict[int, np.ndarray] = {}
        marker_y_axes: dict[int, np.ndarray] = {}
        marker_corners: dict[int, np.ndarray] = {}
        for marker_id, marker_slice in self.marker_slices.items():
            x, y, yaw = parameters[marker_slice]
            center = plane_origin + x * plane_x + y * plane_y
            marker_x = math.cos(yaw) * plane_x + math.sin(yaw) * plane_y
            marker_y = -math.sin(yaw) * plane_x + math.cos(yaw) * plane_y
            corners = (
                center
                + self.object_corners[:, 0, None] * marker_x
                + self.object_corners[:, 1, None] * marker_y
            )
            marker_centers[marker_id] = center
            marker_x_axes[marker_id] = marker_x
            marker_y_axes[marker_id] = marker_y
            marker_corners[marker_id] = corners
        theta = {0: 0.0}
        theta.update(
            {
                frame_key: float(parameters[index])
                for frame_key, index in self.theta_indices.items()
            }
        )
        return DecodedState(
            axis,
            axis_point,
            plane_normal,
            plane_offset,
            marker_centers,
            marker_x_axes,
            marker_y_axes,
            marker_corners,
            theta,
        )

    def projected_corners(
        self, state: DecodedState, observation: MarkerObservation
    ) -> tuple[np.ndarray, np.ndarray]:
        reference_corners = state.marker_corners[observation.marker_id]
        angle = state.theta[observation.frame_key]
        rotation = Rotation.from_rotvec(state.axis * angle).as_matrix()
        camera_points = (
            (reference_corners - state.axis_point) @ rotation.T + state.axis_point
        )
        projected, _ = cv2.projectPoints(
            camera_points,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            self.calibration.camera_matrix,
            self.calibration.distortion,
        )
        return projected.reshape(4, 2), camera_points[:, 2]

    def residual(self, parameters: np.ndarray) -> np.ndarray:
        state = self.decode(parameters)
        residuals = []
        for observation in self.observations:
            projected, depths = self.projected_corners(state, observation)
            delta = projected - observation.corners
            if np.any(depths <= 1e-3) or not np.all(np.isfinite(delta)):
                delta = np.full((4, 2), 1e4, dtype=np.float64)
            residuals.append(delta.reshape(-1))
        return np.concatenate(residuals)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.initial.shape, -np.inf, dtype=np.float64)
        upper = np.full(self.initial.shape, np.inf, dtype=np.float64)
        lower[0:2] = -2.0
        upper[0:2] = 2.0
        lower[4:6] = -2.0
        upper[4:6] = 2.0
        for frame_key, index in self.theta_indices.items():
            margin = math.pi if frame_key >= self.source_frame_count else math.pi / 2.0
            lower[index] = self.initial[index] - margin
            upper[index] = self.initial[index] + margin
        return lower, upper


def _state_as_initial_values(
    state: DecodedState, marker_ids: Iterable[int]
) -> dict[int, tuple[float, float, float]]:
    seed = int(np.argmin(np.abs(state.plane_normal)))
    plane_x, plane_y = _stable_basis(state.plane_normal, seed)
    origin = state.plane_offset * state.plane_normal
    values = {}
    for marker_id in marker_ids:
        local = state.marker_centers[marker_id] - origin
        marker_x = state.marker_x_axes[marker_id]
        values[marker_id] = (
            float(np.dot(local, plane_x)),
            float(np.dot(local, plane_y)),
            math.atan2(float(np.dot(marker_x, plane_y)), float(np.dot(marker_x, plane_x))),
        )
    return values


def _run_bundle(model: _BundleModel, verbose: bool):
    lower, upper = model.bounds()
    return least_squares(
        model.residual,
        model.initial,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=800,
        verbose=2 if verbose else 0,
    )


def _assign_reprojection_errors(
    model: _BundleModel, parameters: np.ndarray
) -> np.ndarray:
    state = model.decode(parameters)
    rms_values = []
    for observation in model.observations:
        projected, _ = model.projected_corners(state, observation)
        errors = np.linalg.norm(projected - observation.corners, axis=1)
        observation.reprojection_errors = errors
        rms_values.append(float(np.sqrt(np.mean(errors * errors))))
    return np.asarray(rms_values, dtype=np.float64)


def _world_geometry(state: DecodedState) -> WorldGeometry:
    denominator = float(np.dot(state.plane_normal, state.axis))
    if abs(denominator) < 1e-6:
        raise EstimationError(
            "The fitted rotation axis is nearly parallel to the reference table "
            "plane, so their intersection is numerically undefined."
        )
    distance = (
        state.plane_offset - float(np.dot(state.plane_normal, state.axis_point))
    ) / denominator
    origin = state.axis_point + distance * state.axis
    world_z = state.axis.copy()

    # Camera-to-origin, projected onto the plane perpendicular to the fitted axis.
    camera_to_origin = origin
    world_x = camera_to_origin - world_z * float(np.dot(world_z, camera_to_origin))
    if np.linalg.norm(world_x) < 1e-8:
        for fallback in np.eye(3):
            world_x = fallback - world_z * float(np.dot(world_z, fallback))
            if np.linalg.norm(world_x) >= 1e-8:
                break
    world_x /= np.linalg.norm(world_x)
    world_y = np.cross(world_z, world_x)
    world_y /= np.linalg.norm(world_y)
    world_x = np.cross(world_y, world_z)
    world_x /= np.linalg.norm(world_x)
    basis = np.column_stack((world_x, world_y, world_z))

    def point_to_world(point: np.ndarray) -> np.ndarray:
        return basis.T @ (point - origin)

    def vector_to_world(vector: np.ndarray) -> np.ndarray:
        return basis.T @ vector

    reference_normal = state.plane_normal.copy()
    # Report the physical normal on the side facing the camera where possible.
    if float(np.dot(reference_normal, -origin)) < 0.0:
        reference_normal = -reference_normal
    reference_normal_world = vector_to_world(reference_normal)
    camera_position = point_to_world(np.zeros(3, dtype=np.float64))
    optical_axis = vector_to_world(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    optical_axis /= np.linalg.norm(optical_axis)
    axis_world = vector_to_world(state.axis)
    axis_point_world = point_to_world(state.axis_point)
    camera_to_axis = float(np.linalg.norm(np.cross(-state.axis_point, state.axis)))
    optical_vs_axis = math.degrees(
        math.acos(np.clip(float(np.dot(optical_axis, axis_world)), -1.0, 1.0))
    )

    # Orient the World-XY normal toward the camera solely for an intuitive signed
    # downward angle. World Z itself remains fixed by the positive rotation sense.
    normal_toward_camera = axis_world.copy()
    if float(np.dot(normal_toward_camera, camera_position)) < 0.0:
        normal_toward_camera = -normal_toward_camera
    downward_world = math.degrees(
        -math.asin(
            np.clip(float(np.dot(optical_axis, normal_toward_camera)), -1.0, 1.0)
        )
    )
    reference_toward_camera = reference_normal_world.copy()
    if float(np.dot(reference_toward_camera, camera_position)) < 0.0:
        reference_toward_camera = -reference_toward_camera
    downward_reference = math.degrees(
        -math.asin(
            np.clip(float(np.dot(optical_axis, reference_toward_camera)), -1.0, 1.0)
        )
    )
    axis_plane = math.degrees(
        math.acos(
            np.clip(abs(float(np.dot(state.axis, state.plane_normal))), -1.0, 1.0)
        )
    )
    return WorldGeometry(
        origin_camera=origin,
        basis_camera=basis,
        axis_world=axis_world,
        axis_point_world=axis_point_world,
        reference_normal_world=reference_normal_world,
        marker_centers_world={
            marker_id: point_to_world(center)
            for marker_id, center in state.marker_centers.items()
        },
        marker_corners_world={
            marker_id: np.stack([point_to_world(point) for point in corners])
            for marker_id, corners in state.marker_corners.items()
        },
        camera_position_world=camera_position,
        optical_axis_world=optical_axis,
        camera_to_axis_distance_mm=camera_to_axis,
        optical_axis_vs_rotation_axis_deg=optical_vs_axis,
        downward_tilt_world_xy_deg=downward_world,
        downward_tilt_reference_plane_deg=downward_reference,
        axis_vs_reference_normal_deg=axis_plane,
    )


def _covariance_and_diagnostics(
    optimizer,
    model: _BundleModel,
    raw_residual: np.ndarray,
    outlier_threshold: float,
) -> tuple[OptimizationDiagnostics, dict[int, float | None]]:
    jacobian = np.asarray(optimizer.jac, dtype=np.float64)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = (
        np.finfo(np.float64).eps
        * max(jacobian.shape)
        * singular[0]
        if singular.size
        else math.inf
    )
    rank = int(np.count_nonzero(singular > tolerance))
    condition = None
    if singular.size and singular[-1] > tolerance:
        condition = float(singular[0] / singular[-1])
    degrees_of_freedom = max(1, raw_residual.size - model.initial.size)
    sigma_squared = float(np.dot(raw_residual, raw_residual) / degrees_of_freedom)
    uncertainties: dict[int, float | None] = {0: 0.0}
    if rank == jacobian.shape[1]:
        covariance = sigma_squared * np.linalg.pinv(jacobian.T @ jacobian, rcond=1e-12)
        for frame_key, parameter_index in model.theta_indices.items():
            variance = float(covariance[parameter_index, parameter_index])
            uncertainties[frame_key] = (
                math.degrees(math.sqrt(max(0.0, variance)))
                if math.isfinite(variance)
                else None
            )
    else:
        uncertainties.update({frame_key: None for frame_key in model.theta_indices})
    diagnostics = OptimizationDiagnostics(
        success=bool(optimizer.success),
        status=int(optimizer.status),
        message=str(optimizer.message),
        cost=float(optimizer.cost),
        evaluations=int(optimizer.nfev),
        optimality=float(optimizer.optimality),
        jacobian_rank=rank,
        jacobian_columns=jacobian.shape[1],
        jacobian_condition=condition,
        outlier_threshold_px=outlier_threshold,
    )
    return diagnostics, uncertainties


def estimate_turntable_geometry(
    frames: list[FrameRecord],
    reference_frame: FrameRecord | None,
    calibration: CameraCalibration,
    marker_size_mm: float,
    dictionary_name: str,
    nominal_step_deg: float,
    verbose: bool = False,
) -> EstimationResult:
    """Estimate the marker map, fixed axis, and one measured angle per solved frame."""
    observations = [item for frame in frames for item in frame.observations]
    if reference_frame is not None:
        observations.extend(reference_frame.observations)
    components = validate_observation_graph(observations, len(frames))
    frame_transforms, marker_transforms = _initial_pose_factorization(
        observations, marker_size_mm
    )
    axis, axis_point, theta = _initial_axis_and_angles(
        frame_transforms, len(frames), math.radians(nominal_step_deg)
    )
    plane_normal, plane_offset, marker_values = _initial_plane_and_layout(
        marker_transforms, marker_size_mm
    )
    model = _BundleModel(
        observations,
        calibration,
        marker_size_mm,
        axis,
        axis_point,
        plane_normal,
        plane_offset,
        marker_values,
        theta,
        len(frames),
    )
    optimizer = _run_bundle(model, verbose)
    rms_values = _assign_reprojection_errors(model, optimizer.x)
    median = float(np.median(rms_values))
    mad = float(np.median(np.abs(rms_values - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(1.5, min(8.0, median + 4.0 * max(robust_sigma, 0.05)))
    for observation, rms in zip(model.observations, rms_values):
        observation.used = bool(rms <= threshold)
        observation.rejection_reason = None if observation.used else "reprojection outlier"

    # If one observation in a frame is bad, keep the good observations. A frame
    # with no surviving observation is honestly unresolved rather than interpolated.
    used_observations = [item for item in observations if item.used]
    used_components = validate_observation_graph(
        used_observations, len(frames), used_only=True
    )
    first_state = model.decode(optimizer.x)
    active_marker_ids = {item.marker_id for item in used_observations}
    refined_model = _BundleModel(
        used_observations,
        calibration,
        marker_size_mm,
        first_state.axis,
        first_state.axis_point,
        first_state.plane_normal,
        first_state.plane_offset,
        _state_as_initial_values(first_state, active_marker_ids),
        first_state.theta,
        len(frames),
    )
    refined = _run_bundle(refined_model, verbose)
    final_rms = _assign_reprojection_errors(refined_model, refined.x)
    final_state = refined_model.decode(refined.x)
    raw_residual = refined_model.residual(refined.x)
    diagnostics, uncertainty = _covariance_and_diagnostics(
        refined, refined_model, raw_residual, threshold
    )
    if not diagnostics.success:
        raise EstimationError(
            "global bundle adjustment did not converge: " + diagnostics.message
        )
    if diagnostics.jacobian_rank < diagnostics.jacobian_columns:
        raise EstimationError(
            "global bundle adjustment is rank-deficient "
            f"({diagnostics.jacobian_rank}/{diagnostics.jacobian_columns}); the "
            "marker visibility or angular span does not constrain a unique solution"
        )
    if not np.all(np.isfinite(final_rms)):
        raise EstimationError("global bundle adjustment produced non-finite residuals")

    solved_frames = {
        observation.frame_key
        for observation in used_observations
        if observation.frame_key < len(frames)
    }
    unresolved = set(range(len(frames))) - solved_frames
    world = _world_geometry(final_state)
    return EstimationResult(
        frames=frames,
        reference_frame=reference_frame,
        calibration=calibration,
        marker_size_mm=marker_size_mm,
        dictionary_name=dictionary_name,
        state=final_state,
        world=world,
        diagnostics=diagnostics,
        graph_components=used_components,
        angle_uncertainty_deg=uncertainty,
        unresolved_frames=unresolved,
    )


def project_camera_points(
    points: np.ndarray, calibration: CameraCalibration
) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        calibration.camera_matrix,
        calibration.distortion,
    )
    return projected.reshape(-1, 2)
