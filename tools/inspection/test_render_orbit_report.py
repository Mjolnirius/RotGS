from argparse import Namespace
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from tools.inspection import render_orbit_report as report
from tools.inspection.render_orbit_report import (
    capped_render_camera,
    capped_render_dimensions,
)
from utils.graphics_utils import getProjectionMatrix


class CappedRenderResolutionTests(unittest.TestCase):
    def test_all_report_sources_fit_exactly_1080(self):
        for size in (1435, 1650, 1651, 1777):
            with self.subTest(source=size):
                self.assertEqual(capped_render_dimensions(size, size, 1080), (1080, 1080))

    def test_smaller_inputs_keep_native_dimensions(self):
        for dimensions in ((640, 640), (720, 480), (641, 539), (1080, 1080)):
            with self.subTest(source=dimensions):
                self.assertEqual(capped_render_dimensions(*dimensions, 1080), dimensions)

    def test_rectangular_inputs_preserve_aspect_ratio(self):
        for source, expected in (
            ((1920, 1080), (1080, 608)),
            ((1080, 1920), (608, 1080)),
            ((800, 1600), (540, 1080)),
        ):
            with self.subTest(source=source):
                self.assertEqual(capped_render_dimensions(*source, 1080), expected)
                self.assertLessEqual(expected[0], source[0])
                self.assertLessEqual(expected[1], source[1])

    def test_invalid_dimensions_are_rejected(self):
        for dimensions in ((1435, 1435, 0), (1435, 1435, 1), (0, 100, 1080)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    capped_render_dimensions(*dimensions)

    def test_resizing_preserves_calibrated_projection_and_camera_position(self):
        width = height = 1435
        fx, fy, cx, cy = 1200.0, 1180.0, 732.0, 701.0
        fovx = 2 * math.atan(width / (2 * fx))
        fovy = 2 * math.atan(height / (2 * fy))
        projection = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy,
            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height,
        ).transpose(0, 1)
        source = SimpleNamespace(
            image_width=width, image_height=height, FoVx=fovx, FoVy=fovy,
            znear=0.01, zfar=100.0, world_view_transform=torch.eye(4),
            full_proj_transform=projection,
        )
        resized = capped_render_camera(source, 1080)
        points = torch.tensor([[0.1, -0.2, 2.0, 1.0], [-0.3, 0.4, 3.0, 1.0]])

        def pixels(camera):
            clip = points @ camera.full_proj_transform
            ndc = clip[:, :2] / clip[:, 3:4]
            return (ndc + 1) * torch.tensor([camera.image_width, camera.image_height]) / 2

        torch.testing.assert_close(pixels(resized), pixels(source) * (1080 / 1435))
        torch.testing.assert_close(resized.camera_center, torch.zeros(3))
        self.assertEqual((resized.image_width, resized.image_height), (1080, 1080))
        self.assertEqual((resized.FoVx, resized.FoVy), (fovx, fovy))


class RenderMediaCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        checkpoint = self.model / "point_cloud/iteration_30000/point_cloud.ply"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"mock checkpoint")
        self.report_dir = self.root / "report"
        self.media_dir = self.report_dir / "media" / self.model.name
        self.media_dir.mkdir(parents=True)
        for name, _, _ in report.TRAJECTORIES:
            for suffix in (".mp4", ".gif", "_poster.jpg"):
                (self.media_dir / (name + suffix)).write_bytes(b"cached media")
        self.settings = dict(render_size=None, resolution_divisor=4, frames=10,
                             fps=30, gif_fps=15, gif_width=480)
        self.previous = dict(iteration=30000, width=640, height=480,
                             render_settings=self.settings)

    def render(self, *, overwrite=False):
        (self.report_dir / "report_manifest.json").write_text(json.dumps({
            "schema_version": 1, "products": {self.model.name: self.previous},
        }))
        config = SimpleNamespace(fixed_camera=True, sh_degree=3, white_background=True)
        camera = SimpleNamespace(image_width=640, image_height=480)
        axis, center = torch.tensor([0., 0., 1.]), torch.zeros(3)
        # Mock only the CUDA/model boundary; exercise actual cache selection,
        # existing-media checks, file removal, and manifest loading.
        render = Mock(return_value={"mp4": "new.mp4", "gif": "new.gif", "poster": "new.jpg"})
        with patch.multiple(
            report,
            load_cfg=Mock(return_value=config),
            GaussianModel=Mock(),
            Scene=Mock(return_value=SimpleNamespace(getTrainCameras=lambda: [camera])),
            camera_frame=Mock(return_value=(axis, center, 5., axis, axis, 30.)),
            torch=SimpleNamespace(tensor=Mock(return_value=torch.zeros(3)),
                                  float32=torch.float32,
                                  cuda=SimpleNamespace(empty_cache=Mock())),
            render_trajectory=render,
        ), redirect_stdout(io.StringIO()):
            product = report.render_model(
                self.model, self.report_dir, iteration=-1, resolution=4,
                frames=10, fps=30, gif_fps=15, gif_width=480,
                overwrite=overwrite,
            )
        self.assertEqual(product["iteration"], 30000)
        return render, product

    def test_matching_checkpoint_and_settings_reuse_existing_media(self):
        render, product = self.render()
        render.assert_not_called()
        self.assertEqual(product["media"]["orbit_front"]["mp4"], "orbit_front.mp4")
        self.assertEqual((self.media_dir / "orbit_front.mp4").read_bytes(), b"cached media")

    def test_new_checkpoint_renders_every_trajectory_despite_existing_media(self):
        self.previous["iteration"] = 5000
        render, _ = self.render()
        self.assertEqual(render.call_count, len(report.TRAJECTORIES))
        self.assertFalse((self.media_dir / "orbit_front.mp4").exists())

    def test_changed_settings_render_every_trajectory(self):
        self.previous["render_settings"] = {**self.settings, "frames": 20}
        render, _ = self.render()
        self.assertEqual(render.call_count, len(report.TRAJECTORIES))

    def test_missing_media_renders_only_the_incomplete_trajectory(self):
        (self.media_dir / "orbit_front.gif").unlink()
        render, _ = self.render()
        render.assert_called_once()
        self.assertEqual(render.call_args.args[0], "orbit_front")

    def test_overwrite_renders_every_trajectory(self):
        render, _ = self.render(overwrite=True)
        self.assertEqual(render.call_count, len(report.TRAJECTORIES))


class ReportPresentationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "preprocessing_metadata.json").write_text(json.dumps({
            "image_count": 73, "output_size": [1435, 1435],
            "segmentation": {"backend": "sam3.1"},
            "undistortion": {"calibration_file": "/calibration/camera_calibration.npz"},
            "angles_degrees": list(range(0, 361, 5)),
        }))
        self.manifest = dict(session_name="Session_04", products={},
                             sort_order="run_number_desc", settings={"render_size": 1080})
        self.output = self.root / "report"
        self.output.mkdir()

    def add_product(self, name, run_id, *, logged_iteration=30000):
        model = self.root / name
        model.mkdir()
        config = Namespace(
            source_path=str(self.dataset), iterations=30000, sh_degree=3,
            axis_mode="bounded_tilt", leverage_TAE=True, wo_tiny=True, wo_flow=True,
            densify_from_iter=500, densify_until_iter=13000,
            densification_interval=100, opacity_reset_interval=3000,
        )
        (model / "cfg_args").write_text(repr(config))
        product = dict(
            name=name, model_path=str(model), iteration=30000, run_log_id=run_id,
            axis=[0., 0.866, 0.5], center=[0., 0., 0.], source_elevation_deg=30.,
            width=1080, height=1080, media={},
            logged_evaluation=dict(iteration=logged_iteration, full_frame_psnr=25.123,
                                   foreground_psnr=23.456, foreground_crop_ssim=0.8123,
                                   test_views=10),
        )
        self.manifest["products"][name] = product
        return model

    def write_report(self):
        report.write_report(self.output, self.manifest)
        return (self.output / "index.html").read_text()

    def test_matching_logged_metrics_are_labelled_without_claiming_external_lpips(self):
        self.add_product("model", "R082")
        page = self.write_report()
        self.assertIn("Full-frame PSNR ↑ (float)", page)
        self.assertIn("25.123 dB", page)
        self.assertIn("23.456 dB", page)
        self.assertIn("0.8123", page)
        self.assertIn("<span>External LPIPS ↓</span><strong>pending</strong>", page)
        self.assertIn("Logged in-memory float validation across 10 held-out views", page)

    def test_logged_metrics_for_another_checkpoint_are_not_displayed(self):
        self.add_product("model", "R082", logged_iteration=5000)
        page = self.write_report()
        self.assertNotIn("25.123 dB", page)
        self.assertNotIn("Full-frame PSNR ↑ (float)", page)
        self.assertIn("External held-out evaluation is pending for this checkpoint.", page)

    def test_external_evaluation_takes_precedence_over_logged_metrics(self):
        model = self.add_product("model", "R082")
        (model / "results.json").write_text(json.dumps({
            "ours_30000": {"PSNR": 30.25, "SSIM": 0.9, "LPIPS": 0.12},
        }))
        (model / "per_view.json").write_text(json.dumps({
            "ours_30000": {"PSNR": {"view.png": 30.25}, "LPIPS": {"view.png": 0.12}},
        }))
        page = self.write_report()
        self.assertIn("<span>Mean PSNR ↑</span><strong>30.25 dB</strong>", page)
        self.assertIn("Mean LPIPS ↓", page)
        self.assertNotIn("25.123 dB", page)
        self.assertNotIn("Full-frame PSNR ↑ (float)", page)

    def test_run_ids_sort_numerically_and_supply_search_and_deep_links(self):
        self.add_product("a_older", "R9")
        self.add_product("z_newer", "R82")
        page = self.write_report()
        self.assertLess(page.index('id="R82"'), page.index('id="R9"'))
        self.assertIn('data-search="r82 z_newer"', page)
        self.assertIn("R82 · z_newer", page)

    def test_timestamps_use_zurich_offsets_in_summer_and_winter(self):
        for month, expected in [(7, "14:00:00 CEST"), (1, "13:00:00 CET")]:
            with self.subTest(month=month):
                value = datetime(2026, month, 15, 12, tzinfo=timezone.utc)
                self.assertTrue(report.format_timestamp(value).endswith(expected))


if __name__ == "__main__":
    unittest.main()
