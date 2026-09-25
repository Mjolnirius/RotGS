from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tools.calibration.turntable_angles import turntable_geometry as geometry
from tools.calibration.turntable_angles.turntable_reports import (
    build_frame_rows,
    build_summary,
)


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def _synthetic_problem():
    rng = np.random.default_rng(20260922)
    image_size = (1600, 1200)
    camera_matrix = np.array(
        [[1250.0, 0.0, 800.0], [0.0, 1240.0, 600.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    calibration = geometry.CameraCalibration(
        camera_matrix,
        np.array([-0.035, 0.012, 0.0004, -0.0003, 0.0], dtype=np.float64),
        image_size,
        Path("synthetic_camera_calibration.npz"),
    )
    marker_size = 34.0
    axis = _unit(np.array([0.025, -0.78, -0.625], dtype=np.float64))
    centre = np.array([15.0, 85.0, 880.0], dtype=np.float64)
    tilt_axis = _unit(np.cross(axis, np.array([1.0, 0.1, 0.0])))
    normal = Rotation.from_rotvec(tilt_axis * math.radians(1.25)).apply(axis)
    plane_x, plane_y = geometry._stable_basis(normal, int(np.argmin(np.abs(normal))))
    object_corners = geometry.marker_object_corners(marker_size)
    marker_reference: dict[int, np.ndarray] = {}
    marker_centers: dict[int, np.ndarray] = {}
    for marker_id in range(6):
        phase = 2.0 * math.pi * marker_id / 6.0
        radius = 112.0 if marker_id % 2 == 0 else 82.0
        marker_center = centre + radius * (
            math.cos(phase) * plane_x + math.sin(phase) * plane_y
        )
        yaw = phase + 0.17 * marker_id
        marker_x = math.cos(yaw) * plane_x + math.sin(yaw) * plane_y
        marker_y = -math.sin(yaw) * plane_x + math.cos(yaw) * plane_y
        marker_reference[marker_id] = (
            marker_center
            + object_corners[:, 0, None] * marker_x
            + object_corners[:, 1, None] * marker_y
        )
        marker_centers[marker_id] = marker_center

    nominal_step = 5.0
    step_errors = 0.11 * np.sin(np.linspace(0.0, 8.0 * math.pi, 72))
    step_errors += 0.025 * np.cos(np.linspace(0.0, 5.0 * math.pi, 72))
    angle_deg = np.concatenate(([0.0], np.cumsum(nominal_step + step_errors)))
    angles = np.radians(angle_deg)
    axis_point = centre - axis * float(np.dot(axis, centre))

    frames: list[geometry.FrameRecord] = []
    corrupted = (10, 4)
    for frame_index, angle in enumerate(angles):
        rotation = Rotation.from_rotvec(axis * angle).as_matrix()
        visible = set(range(6)) if frame_index == 0 else {
            (frame_index // 3 + offset) % 6 for offset in range(4)
        }
        observations = []
        for marker_id in sorted(visible):
            camera_points = (
                (marker_reference[marker_id] - axis_point) @ rotation.T + axis_point
            )
            corners = geometry.project_camera_points(camera_points, calibration)
            corners += rng.normal(0.0, 0.055, size=corners.shape)
            if (frame_index, marker_id) == corrupted:
                corners += np.array(
                    [[14.0, -9.0], [-11.0, 13.0], [12.0, 11.0], [-13.0, -10.0]]
                )
            candidates = geometry._pose_candidates(corners, calibration, marker_size)
            if not candidates:
                raise AssertionError("synthetic IPPE initialization unexpectedly failed")
            observations.append(
                geometry.MarkerObservation(
                    frame_key=frame_index,
                    marker_id=marker_id,
                    corners=corners,
                    perimeter_px=float(cv2.arcLength(corners.astype(np.float32), True)),
                    candidates=candidates,
                )
            )
        frames.append(
            geometry.FrameRecord(
                index=frame_index,
                path=Path(f"{frame_index:04d}.png"),
                filename=f"{frame_index:04d}.png",
                observations=observations,
                image_size=image_size,
            )
        )
    truth = {
        "axis": axis,
        "axis_point": axis_point,
        "centre": centre,
        "normal": normal,
        "angles_deg": angle_deg,
        "nominal_step": nominal_step,
        "corrupted": corrupted,
    }
    return frames, calibration, marker_size, truth


class SyntheticGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frames, calibration, marker_size, truth = _synthetic_problem()
        cls.frames = frames
        cls.calibration = calibration
        cls.marker_size = marker_size
        cls.truth = truth
        cls.result = geometry.estimate_turntable_geometry(
            frames,
            None,
            calibration,
            marker_size,
            "DICT_5X5_50",
            truth["nominal_step"],
        )

    def test_recovers_nonuniform_unwrapped_angles_and_closure(self):
        measured = np.array(
            [math.degrees(self.result.state.theta[index]) for index in range(len(self.truth["angles_deg"]))]
        )
        np.testing.assert_allclose(measured, self.truth["angles_deg"], atol=0.16)
        self.assertGreater(measured[-1], 359.0)
        expected_closure = self.truth["angles_deg"][-1] - (len(self.truth["angles_deg"]) - 1) * self.truth["nominal_step"]
        measured_closure = measured[-1] - (len(measured) - 1) * self.truth["nominal_step"]
        self.assertAlmostEqual(measured_closure, expected_closure, delta=0.16)
        measured_steps = np.diff(measured)
        true_steps = np.diff(self.truth["angles_deg"])
        np.testing.assert_allclose(measured_steps, true_steps, atol=0.22)

    def test_recovers_axis_centre_plane_and_camera_geometry(self):
        self.assertGreater(float(np.dot(self.result.state.axis, self.truth["axis"])), 0.999)
        np.testing.assert_allclose(
            self.result.world.origin_camera, self.truth["centre"], atol=2.5
        )
        expected_tilt = math.degrees(
            math.acos(
                np.clip(
                    abs(float(np.dot(self.truth["axis"], self.truth["normal"]))),
                    -1.0,
                    1.0,
                )
            )
        )
        self.assertAlmostEqual(
            self.result.world.axis_vs_reference_normal_deg, expected_tilt, delta=0.35
        )
        expected_distance = np.linalg.norm(
            np.cross(-self.truth["axis_point"], self.truth["axis"])
        )
        self.assertAlmostEqual(
            self.result.world.camera_to_axis_distance_mm,
            expected_distance,
            delta=2.0,
        )
        expected_optical_angle = math.degrees(
            math.acos(np.clip(float(self.truth["axis"][2]), -1.0, 1.0))
        )
        self.assertAlmostEqual(
            self.result.world.optical_axis_vs_rotation_axis_deg,
            expected_optical_angle,
            delta=0.25,
        )
        self.assertTrue(math.isfinite(self.result.world.downward_tilt_world_xy_deg))
        self.assertTrue(
            math.isfinite(self.result.world.downward_tilt_reference_plane_deg)
        )

    def test_rejects_corrupted_observation(self):
        frame_index, marker_id = self.truth["corrupted"]
        observation = next(
            item
            for item in self.frames[frame_index].observations
            if item.marker_id == marker_id
        )
        self.assertFalse(observation.used)
        self.assertEqual(observation.rejection_reason, "reprojection outlier")

    def test_rows_and_summary_preserve_physical_measurements(self):
        rows = build_frame_rows(self.result, self.truth["nominal_step"], 0.0)
        summary = build_summary(self.result, rows, self.truth["nominal_step"], 0.0)
        self.assertEqual(rows[0]["estimated_angle_deg"], 0.0)
        self.assertIsNone(rows[0]["step_deg"])
        self.assertAlmostEqual(
            rows[-1]["estimated_angle_deg"], self.truth["angles_deg"][-1], delta=0.16
        )
        self.assertAlmostEqual(
            summary["angle_results"]["closure_error_deg"],
            self.truth["angles_deg"][-1] - (len(self.truth["angles_deg"]) - 1) * self.truth["nominal_step"],
            delta=0.16,
        )
        self.assertEqual(
            summary["geometry"]["world_frame"]["y_axis"], "World Z cross World X"
        )
        self.assertEqual(len(summary["geometry"]["table_normal_by_frame_world"]), len(self.truth["angles_deg"]))


class ValidationTests(unittest.TestCase):
    def test_natural_filename_order_does_not_parse_angles(self):
        paths = [Path("item_10.png"), Path("item_2.png"), Path("item_001.png")]
        ordered = sorted(paths, key=geometry.natural_sort_key)
        self.assertEqual([item.name for item in ordered], ["item_001.png", "item_2.png", "item_10.png"])

    def test_disconnected_observation_graph_fails(self):
        observations = [
            geometry.MarkerObservation(0, 1, np.zeros((4, 2)), 1.0),
            geometry.MarkerObservation(1, 1, np.zeros((4, 2)), 1.0),
            geometry.MarkerObservation(2, 2, np.zeros((4, 2)), 1.0),
            geometry.MarkerObservation(3, 2, np.zeros((4, 2)), 1.0),
        ]
        with self.assertRaisesRegex(geometry.EstimationError, "disconnected"):
            geometry.validate_observation_graph(observations, 4)

    def test_insufficient_unique_markers_fails(self):
        observations = [
            geometry.MarkerObservation(index, 7, np.zeros((4, 2)), 1.0)
            for index in range(3)
        ]
        with self.assertRaisesRegex(geometry.EstimationError, "two unique"):
            geometry.validate_observation_graph(observations, 3)

    def test_invalid_intrinsics_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            np.savez(path, camera_matrix=np.eye(2), dist_coeffs=np.zeros(5), image_size=[640, 480])
            with self.assertRaisesRegex(ValueError, "3x3"):
                geometry.load_calibration(path)

    def test_incorrect_image_dimensions_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.png"
            cv2.imwrite(str(path), np.full((240, 320, 3), 255, dtype=np.uint8))
            calibration = geometry.CameraCalibration(
                np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]),
                np.zeros(5),
                (640, 480),
                Path("synthetic.npz"),
            )
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                geometry.detect_frames([path], calibration, 40.0, "DICT_5X5_50")

    def test_aruco_detection_and_subpixel_corners(self):
        with tempfile.TemporaryDirectory() as temporary:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            marker = cv2.aruco.generateImageMarker(dictionary, 7, 220)
            canvas = np.full((600, 800), 255, dtype=np.uint8)
            canvas[190:410, 290:510] = marker
            path = Path(temporary) / "marker.png"
            cv2.imwrite(str(path), canvas)
            calibration = geometry.CameraCalibration(
                np.array([[900.0, 0, 400.0], [0, 900.0, 300.0], [0, 0, 1.0]]),
                np.zeros(5),
                (800, 600),
                Path("synthetic.npz"),
            )
            frames, _ = geometry.detect_frames(
                [path], calibration, 40.0, "DICT_5X5_50"
            )
            self.assertEqual([item.marker_id for item in frames[0].observations], [7])
            self.assertEqual(frames[0].observations[0].corners.shape, (4, 2))
            self.assertGreaterEqual(len(frames[0].observations[0].candidates), 1)


if __name__ == "__main__":
    unittest.main()
