"""Tests for metadata-relative multi-camera tilt bounds."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from scene.gaussian_model import GaussianModel


def _axis_at(tilt_degrees: float, side_degrees: float = 0.0) -> torch.Tensor:
    tilt = math.radians(tilt_degrees)
    side = math.radians(side_degrees)
    return torch.tensor(
        [
            math.sin(side),
            math.cos(side) * math.cos(tilt),
            math.cos(side) * math.sin(tilt),
        ],
        dtype=torch.float32,
    )


def _tilt_degrees(axis: torch.Tensor) -> float:
    return math.degrees(math.atan2(float(axis[2]), float(axis[1])))


def _side_degrees(axis: torch.Tensor) -> float:
    return math.degrees(math.asin(float(axis[0])))


class MetadataRelativeTiltTests(unittest.TestCase):
    def make_model(self, deviation=5.0) -> GaussianModel:
        return GaussianModel(
            3,
            multi_camera=True,
            number_of_cameras=3,
            axis_mode="bounded_tilt",
            axis_tilt_init_degrees=(5.0, 30.0, 60.0),
            axis_tilt_deviation_limit_deg=deviation,
            axis_side_limit_deg=5.0,
        )

    def test_each_camera_uses_its_own_metadata_centered_bound(self) -> None:
        model = self.make_model()
        model._axis = torch.stack([_axis_at(40.0)] * 3)

        tilts = [_tilt_degrees(model.get_axis(index)) for index in range(3)]

        self.assertAlmostEqual(tilts[0], 10.0, places=4)
        self.assertAlmostEqual(tilts[1], 35.0, places=4)
        self.assertAlmostEqual(tilts[2], 55.0, places=4)

    def test_side_remains_independently_bounded(self) -> None:
        model = self.make_model()
        model._axis[1] = _axis_at(30.0, side_degrees=20.0)

        axis = model.get_axis(1)

        self.assertAlmostEqual(_tilt_degrees(axis), 30.0, places=4)
        self.assertAlmostEqual(_side_degrees(axis), 5.0, places=4)

    def test_zero_deviation_fixes_metadata_tilts(self) -> None:
        model = self.make_model(deviation=0.0)
        model._axis = torch.stack([_axis_at(45.0)] * 3)

        tilts = [_tilt_degrees(model.get_axis(index)) for index in range(3)]

        for actual, expected in zip(tilts, (5.0, 30.0, 60.0)):
            self.assertAlmostEqual(actual, expected, places=4)

    def test_deviation_limit_requires_metadata_tilts(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires per-camera"):
            GaussianModel(
                3,
                axis_mode="bounded_tilt",
                axis_tilt_deviation_limit_deg=5.0,
            )


class MotionInitializationTests(unittest.TestCase):
    def test_loads_motion_without_geometry_and_recenters_priors(self) -> None:
        model = GaussianModel(
            3,
            multi_camera=True,
            number_of_cameras=3,
            freeze_axis=True,
            freeze_center=True,
            freeze_depth=True,
            axis_mode="bounded_tilt",
            axis_tilt_init_degrees=(5.0, 30.0, 60.0),
            axis_tilt_deviation_limit_deg=5.0,
            depth_reference_camera_index=1,
            multi_camera_transform="rigid",
        )
        axes = torch.stack(
            [_axis_at(9.0, 0.1), _axis_at(31.0, 0.2), _axis_at(64.0, 0.3)]
        )
        lateral_centers = torch.tensor(
            [[0.1, 0.2, 9.0], [0.0, 0.3, 8.0], [-0.1, 0.4, 7.0]]
        )
        depths = torch.tensor([0.4, 0.0, -0.3])
        payload = {
            "multi_camera_transform": "rigid",
            "axis": axes.tolist(),
            # Deliberately omit the legacy combined center field: current
            # motion.json files have a dedicated lateral center.
            "lateral_center": lateral_centers.tolist(),
            "camera_depth": depths.tolist(),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            motion_path = Path(temporary_directory) / "motion.json"
            motion_path.write_text(json.dumps(payload), encoding="utf-8")
            model.load_motion_from_json(motion_path)

        self.assertTrue(torch.allclose(model._axis, axes))
        expected_lateral = lateral_centers.clone()
        expected_lateral[:, 2] = 0.0
        self.assertTrue(torch.allclose(model._center_initial, expected_lateral))
        self.assertTrue(torch.allclose(model._center_point, expected_lateral))
        actual_depths = torch.stack(
            [model.get_camera_depth(index) for index in range(3)]
        )
        self.assertTrue(torch.allclose(actual_depths, depths, atol=1e-6))
        self.assertFalse(model._axis.requires_grad)
        self.assertFalse(model._center_point.requires_grad)
        self.assertFalse(model._camera_depth.requires_grad)


class DensificationLimitTests(unittest.TestCase):
    def test_limit_keeps_highest_scores(self) -> None:
        selected = torch.tensor([True, False, True, True, False])
        scores = torch.tensor([0.1, 9.0, 0.8, 0.4, 8.0])

        limited = GaussianModel._limit_densification_mask(
            selected, scores, max_selected=2
        )

        self.assertTrue(torch.equal(
            limited, torch.tensor([False, False, True, True, False])
        ))

    def test_zero_limit_disables_new_selections(self) -> None:
        selected = torch.tensor([True, False, True])
        scores = torch.tensor([0.1, 0.2, 0.3])

        limited = GaussianModel._limit_densification_mask(
            selected, scores, max_selected=0
        )

        self.assertFalse(limited.any())


if __name__ == "__main__":
    unittest.main()
