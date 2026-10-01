from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from scene.residual_predictor import ResidualPredictor
from scene.tae_calibration import (
    configure_tae_training_args,
    load_tae_calibration,
)


class TAECalibrationTest(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        tae_dir = dataset / "turntable_angle_estimation"
        tae_dir.mkdir(parents=True)
        (dataset / "preprocessing_metadata.json").write_text(
            json.dumps(
                {
                    "renamed_images": [
                        {
                            "source": "source_0.png",
                            "angle_degrees": 0.0,
                            "output": "0000.png",
                        },
                        {
                            "source": "source_5.png",
                            "angle_degrees": 5.0,
                            "output": "0001.png",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (tae_dir / "summary.json").write_text(
            json.dumps(
                {
                    "input": {"image_count": 2},
                    "geometry": {
                        "rotation_centre_camera_mm": [-4.0, 90.0, 570.0],
                        "world_frame": {
                            "basis_vectors_in_opencv_camera_coordinates": {
                                "z": [0.005, 0.866, 0.5]
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        with (tae_dir / "turntable_angles.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=(
                    "frame_index",
                    "filename",
                    "nominal_angle_deg",
                    "estimated_angle_deg",
                    "angle_uncertainty_deg",
                    "confidence",
                ),
            )
            writer.writeheader()
            writer.writerows(
                (
                    {
                        "frame_index": 0,
                        "filename": "source_0.png",
                        "nominal_angle_deg": 0.0,
                        "estimated_angle_deg": 0.1,
                        "angle_uncertainty_deg": 0.01,
                        "confidence": "good",
                    },
                    {
                        "frame_index": 1,
                        "filename": "source_5.png",
                        "nominal_angle_deg": 5.0,
                        "estimated_angle_deg": 5.05,
                        "angle_uncertainty_deg": 0.02,
                        "confidence": "good",
                    },
                )
            )
        return dataset

    def test_loads_and_joins_preprocessed_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            calibration = load_tae_calibration(
                self.make_dataset(Path(temporary)),
                image_names=["0000.png", "0001.png"],
            )
        self.assertEqual(calibration.angles_by_image_deg["0001.png"], 5.05)
        self.assertAlmostEqual(calibration.measured_total_rotation_deg, 4.95)
        self.assertAlmostEqual(calibration.maximum_angle_uncertainty_deg, 0.02)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in calibration.axis_camera)), 1.0)

    def test_configures_uncertainty_bound_and_learnable_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary))
            args = Namespace(
                multi_camera=False,
                angle_noise_std=0.0,
                wo_tiny=False,
                max_residual_angle_deg=0.0,
                max_sweep_error_deg=0.0,
                center_max_offset=0.25,
            )
            calibration = configure_tae_training_args(args, dataset)
        self.assertEqual(args.axis_mode, "bounded_tilt")
        self.assertAlmostEqual(args.axis_tilt_min_deg, args.axis_tilt_init_deg - 0.5)
        self.assertAlmostEqual(args.axis_tilt_max_deg, args.axis_tilt_init_deg + 0.5)
        self.assertAlmostEqual(args.axis_side_init_deg, calibration.axis_side_deg)
        self.assertEqual(args.axis_side_limit_deg, 0.5)
        self.assertEqual(args.max_residual_angle_deg, 0.06)
        self.assertEqual(args.max_sweep_error_deg, 0.15)
        self.assertEqual(args.center_max_offset, 0.05)

    def test_tae_local_correction_is_bounded_and_endpoint_anchored(self):
        predictor = ResidualPredictor(
            1,
            num_ctrl_points=8,
            device="cpu",
            max_residual_angle_deg=0.05,
            max_sweep_error_deg=0.15,
            anchor_local_endpoints=True,
        )
        predictor.residuals.data.fill_(10.0)
        self.assertEqual(
            float(predictor(torch.tensor(0.0), 0).detach()), 0.0
        )
        self.assertEqual(
            float(predictor(torch.tensor(1.0), 0).detach()), 0.0
        )
        midpoint_deg = math.degrees(
            float(predictor(torch.tensor(0.5), 0).detach())
        )
        self.assertLessEqual(midpoint_deg, 0.050001)


if __name__ == "__main__":
    unittest.main()
