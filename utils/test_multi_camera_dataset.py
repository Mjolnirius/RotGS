"""Tests for multi-camera bundle discovery and trainer-facing helpers."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from utils.multi_camera_dataset import (
    axis_vectors_from_elevations,
    group_cameras_by_index,
    load_multi_camera_bundle,
)


def _make_camera_directory(root: Path, name: str) -> Path:
    camera = root / name
    (camera / "images").mkdir(parents=True)
    (camera / "images" / "frame_000.png").touch()
    sparse = camera / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 100 100 80 80 50 50\n", encoding="utf-8"
    )
    return camera


class MultiCameraBundleTests(unittest.TestCase):
    def test_metadata_controls_semantic_order_and_elevations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_camera_directory(root, "camera_high")
            _make_camera_directory(root, "camera_low")
            metadata = {
                "dataset_type": "rotgs_multi_camera",
                "camera_count": 2,
                "rough_camera_elevations_degrees": [5.0, 60.0],
                "cameras": [
                    {
                        "camera_index": 1,
                        "directory": "camera_high",
                        "rough_elevation_degrees": 60.0,
                    },
                    {
                        "camera_index": 0,
                        "directory": "camera_low",
                        "rough_elevation_degrees": 5.0,
                    },
                ],
            }
            (root / "multi_camera_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            bundle = load_multi_camera_bundle(root)

        self.assertEqual(
            [camera.directory_name for camera in bundle.passes],
            ["camera_low", "camera_high"],
        )
        self.assertEqual(bundle.rough_elevations_degrees, (5.0, 60.0))

    def test_legacy_directories_remain_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_camera_directory(root, "cam_b")
            _make_camera_directory(root, "cam_a")

            bundle = load_multi_camera_bundle(root)

        self.assertEqual(
            [camera.directory_name for camera in bundle.passes],
            ["cam_a", "cam_b"],
        )
        self.assertIsNone(bundle.metadata_path)
        self.assertIsNone(bundle.rough_elevations_degrees)

    def test_rejects_inconsistent_top_level_elevations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_camera_directory(root, "cam_0")
            _make_camera_directory(root, "cam_1")
            metadata = {
                "dataset_type": "rotgs_multi_camera",
                "camera_count": 2,
                "rough_camera_elevations_degrees": [5.0, 30.0],
                "cameras": [
                    {
                        "camera_index": 0,
                        "directory": "cam_0",
                        "rough_elevation_degrees": 5.0,
                    },
                    {
                        "camera_index": 1,
                        "directory": "cam_1",
                        "rough_elevation_degrees": 60.0,
                    },
                ],
            }
            (root / "multi_camera_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "must match camera entries"):
                load_multi_camera_bundle(root)


class TrainerHelperTests(unittest.TestCase):
    def test_elevations_map_to_normalized_yz_axes(self) -> None:
        axes = axis_vectors_from_elevations([0.0, 30.0, 90.0])
        self.assertEqual(axes[0], (0.0, 1.0, 0.0))
        self.assertTrue(math.isclose(axes[1][1], math.sqrt(3) / 2))
        self.assertTrue(math.isclose(axes[1][2], 0.5))
        self.assertTrue(math.isclose(axes[2][1], 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(axes[2][2], 1.0))

    def test_groups_interleaved_views_by_camera_index(self) -> None:
        @dataclass
        class Camera:
            name: str
            cam_idx: int

        cameras = [Camera("b0", 1), Camera("a0", 0), Camera("b1", 1)]

        grouped = group_cameras_by_index(cameras, 2)

        self.assertEqual(
            [[camera.name for camera in group] for group in grouped],
            [["a0"], ["b0", "b1"]],
        )

    def test_rejects_missing_camera_views(self) -> None:
        @dataclass
        class Camera:
            cam_idx: int

        with self.assertRaisesRegex(ValueError, r"camera indices \[1\]"):
            group_cameras_by_index([Camera(0)], 2)


if __name__ == "__main__":
    unittest.main()
