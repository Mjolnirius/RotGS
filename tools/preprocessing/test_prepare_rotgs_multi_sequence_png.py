"""Focused tests for multi-camera PNG preprocessing orchestration."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tools.preprocessing import prepare_rotgs_multi_sequence_png as multi


class CameraAngleParsingTests(unittest.TestCase):
    def test_parses_final_degree_token_and_product_stem(self) -> None:
        self.assertEqual(
            multi._parse_camera_angle("Quoellfrisch_top_5d"),
            (5.0, "Quoellfrisch_top"),
        )
        self.assertEqual(
            multi._parse_camera_angle("product_-12.5deg"),
            (-12.5, "product"),
        )

    def test_resolves_and_sorts_camera_passes_by_elevation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folders = [
                root / "can_top_55d",
                root / "can_top_5d",
                root / "can_top_30d",
            ]
            for folder in folders:
                folder.mkdir()

            passes = multi._resolve_camera_passes(folders)

        self.assertEqual(
            [item.rough_elevation_degrees for item in passes],
            [5.0, 30.0, 55.0],
        )
        self.assertTrue(all(item.product_stem == "can_top" for item in passes))

    def test_rejects_mismatched_product_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "can_top_5d"
            second = root / "bottle_top_30d"
            first.mkdir()
            second.mkdir()

            with self.assertRaisesRegex(ValueError, "one common product"):
                multi._resolve_camera_passes([first, second])


class OutputNamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_passes = [
            multi.CameraPass(Path("/tmp/can_top_5d"), 5.0, "can_top"),
            multi.CameraPass(Path("/tmp/can_top_30d"), 30.0, "can_top"),
        ]

    def test_square_regenerated_alpha_name(self) -> None:
        self.assertEqual(
            multi._default_output_name(
                self.camera_passes,
                overwrite_alpha_mask=True,
                square_crop=True,
                output_sizes=[(1777, 1777), (1777, 1777)],
            ),
            "can_top_multi_5d_30d_und_rn_roi_sqr_PNGaR_1777",
        )

    def test_source_aspect_mixed_resolution_name(self) -> None:
        self.assertEqual(
            multi._default_output_name(
                self.camera_passes,
                overwrite_alpha_mask=False,
                square_crop=False,
                output_sizes=[(1777, 1200), (1600, 1100)],
            ),
            "can_top_multi_5d_30d_und_rn_roi_PNGa_ds_mixedres",
        )


class MultiCameraBundleTests(unittest.TestCase):
    def test_builds_sorted_atomic_bundle_and_handoff_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folders = [
                root / "can_top_30d",
                root / "can_top_5d",
                root / "can_top_55d",
            ]
            for folder in source_folders:
                folder.mkdir()
            source_cameras = root / "cameras.txt"
            source_cameras.write_text(
                "1 PINHOLE 200 200 100 100 100 100\n",
                encoding="utf-8",
            )
            calibration = root / "camera_calibration.npz"
            calibration.touch()

            def fake_prepare(
                folder_name: str | Path,
                cameras_file: str | Path,
                **kwargs: object,
            ) -> dict[str, object]:
                destination = Path(kwargs["destination_folder"])
                (destination / "images").mkdir(parents=True)
                sparse = destination / "sparse" / "0"
                sparse.mkdir(parents=True)
                (sparse / "cameras.txt").write_text(
                    "1 PINHOLE 120 120 60 60 60 60\n",
                    encoding="utf-8",
                )
                metadata = {
                    "image_count": 3,
                    "angles_degrees": [0.0, 5.0, 10.0],
                    "source_size": [200, 200],
                    "output_size": [120, 120],
                    "product_selection_left_top_right_bottom": [20, 20, 180, 180],
                    "crop_box_left_top_right_bottom": [20, 20, 180, 180],
                    "undistortion": {"calibration_file": str(calibration)},
                }
                (destination / "preprocessing_metadata.json").write_text(
                    json.dumps(metadata),
                    encoding="utf-8",
                )
                return {
                    "input_folder": str(folder_name),
                    "output_folder": str(destination),
                    "output_cameras_file": str(sparse / "cameras.txt"),
                    "metadata_file": str(
                        destination / "preprocessing_metadata.json"
                    ),
                }

            with (
                mock.patch.object(
                    multi,
                    "prepare_rotgs_sequence_png",
                    side_effect=fake_prepare,
                ) as mocked_prepare,
                redirect_stdout(io.StringIO()),
            ):
                summary = multi.prepare_rotgs_multi_sequence_png(
                    source_folders,
                    source_cameras,
                    overwrite_alpha_mask=True,
                    skip_source_validation=True,
                )

            output = Path(summary["output_folder"])
            self.assertTrue(output.is_dir())
            self.assertEqual(
                [path.name for path in sorted(output.iterdir()) if path.is_dir()],
                ["camera_00_5d", "camera_01_30d", "camera_02_55d"],
            )
            metadata = json.loads(
                (output / "multi_camera_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                metadata["rough_camera_elevations_degrees"],
                [5.0, 30.0, 55.0],
            )
            self.assertTrue(metadata["source_validation_skipped"])
            self.assertTrue(
                metadata["trainer_handoff"][
                    "legacy_train_multi_single_rasterizer_compatible"
                ]
            )
            self.assertEqual(mocked_prepare.call_count, 3)
            self.assertTrue(
                all(
                    call.kwargs["skip_source_validation"]
                    for call in mocked_prepare.call_args_list
                )
            )
            labels = [
                call.kwargs["review_label"]
                for call in mocked_prepare.call_args_list
            ]
            self.assertIn("rough elevation 5°", labels[0])
            self.assertIn("rough elevation 55°", labels[2])
            for camera_summary in summary["camera_summaries"]:
                self.assertTrue(
                    Path(camera_summary["output_folder"]).is_relative_to(output)
                )


if __name__ == "__main__":
    unittest.main()
