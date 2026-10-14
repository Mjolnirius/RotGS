"""Check angle recovery against training and transform recovery against rendering."""
from __future__ import annotations

from argparse import Namespace
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.inspection.export_view_poses import extract_view_poses, rigid_transform
from scene.residual_predictor import ResidualPredictor
from utils.general_utils import axis_angle2rotmat, rotate_vector_by_quaternion


class ViewPoseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_run(self, **overrides):
        dataset, run = self.root / 'dataset', self.root / 'run'
        images = dataset / 'images'
        images.mkdir(parents=True)
        renamed = []
        for i in range(73):
            name = f'{i:04d}.png'
            (images / name).touch()
            renamed.append({'source': f'source_{i * 5}.png', 'output': name, 'angle_degrees': i * 5})
        (dataset / 'preprocessing_metadata.json').write_text(json.dumps({'renamed_images': renamed}))
        cfg = Namespace(source_path=str(dataset), fixed_camera=True, multi_camera=False,
                        images='images', eval=True, iterations=30000, rotation_direction=1,
                        leverage_TAE=False, wo_tiny=False, max_residual_angle_deg=0.0,
                        max_sweep_error_deg=4.0, angle_noise_std=0.0,
                        checkpoint_iterations=[], test_iterations=[])
        vars(cfg).update(overrides)
        run.mkdir()
        (run / 'cfg_args').write_text(repr(cfg))
        snapshot = run / 'point_cloud/iteration_30000'
        snapshot.mkdir(parents=True)
        np.save(snapshot / 'axis.npy', [[0., math.cos(math.pi / 6), math.sin(math.pi / 6)]])
        np.save(snapshot / 'center.npy', [[0.1, 0.4, -0.02]])
        predictor = ResidualPredictor(device='cpu', num_ctrl_points=8 if cfg.leverage_TAE else 1,
                                      max_residual_angle_deg=cfg.max_residual_angle_deg,
                                      max_sweep_error_deg=cfg.max_sweep_error_deg,
                                      anchor_local_endpoints=cfg.leverage_TAE)
        predictor.save_weights(str(run), 30000)
        return run, dataset, predictor, cfg

    def write_aruco(self, dataset, rows):
        folder = dataset / 'turntable_angle_estimation'
        folder.mkdir(exist_ok=True)
        with (folder / 'turntable_angles.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['frame_index', 'filename', 'nominal_angle_deg',
                                                       'estimated_angle_deg', 'angle_uncertainty_deg', 'confidence'])
            writer.writeheader()
            writer.writerows(rows)
        return folder

    def test_transform_matches_renderer_quaternion_rotation(self):
        axis = torch.tensor([0.2, 0.9, -0.35])
        axis /= axis.norm()
        center = torch.tensor([0.3, -0.5, 0.7])
        points = torch.tensor([[1., 2., 3.], [-0.8, 0.2, 0.1], [0.3, -0.5, 0.7]])
        for angle in [0., 0.8, -3.7, 2 * math.pi + 0.01]:
            theta = torch.tensor([angle])
            quaternion = torch.cat([torch.cos(theta / 2), axis * torch.sin(theta / 2)])
            rendered = rotate_vector_by_quaternion(points - center, quaternion.repeat(len(points), 1)) + center
            transform = rigid_transform(axis.numpy(), center.numpy(), angle)
            exported = points.numpy() @ transform[:3, :3].T + transform[:3, 3]
            np.testing.assert_allclose(exported, rendered.numpy(), atol=1e-6)
            np.testing.assert_allclose(transform[:3, :3], axis_angle2rotmat(axis, theta).numpy(), atol=3e-7)
            np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1)

    def test_all_73_views_preserve_full_turn_and_held_out_frames(self):
        run, _, _, _ = self.make_run()
        report = extract_view_poses(run)
        self.assertEqual(len(report['views']), 73)
        self.assertEqual(sum(row['split'] == 'test' for row in report['views']), 10)
        self.assertEqual(report['views'][-1]['image_name'], '0072.png')
        self.assertAlmostEqual(report['total_rotation_deg'], 360, places=4)
        self.assertEqual(report['views'][0]['rotgs_angle_deg'], 0)

    def test_wo_tiny_still_applies_sweep_and_negative_direction(self):
        run, _, predictor, _ = self.make_run(wo_tiny=True, rotation_direction=-1)
        predictor.residuals.data.fill_(2)  # Disabled local correction must be ignored.
        predictor.sweep_error.data.fill_(0.02)
        predictor.save_weights(str(run), 30000)
        report = extract_view_poses(run)
        expected = 360 - math.degrees(float(predictor.effective_sweep_error(0).detach()))
        self.assertAlmostEqual(report['total_rotation_deg'], expected, places=4)
        self.assertTrue(all(row['local_correction_deg'] == 0 for row in report['views']))
        self.assertLess(report['views'][-1]['rotgs_render_angle_deg'], -350)

    def test_raw_zero_and_camera_extrinsics_are_retained(self):
        run, _, predictor, _ = self.make_run(max_sweep_error_deg=0)
        predictor.residuals.data.copy_(torch.tensor([[-0.4, -0.3]]))
        predictor.save_weights(str(run), 30000)
        report = extract_view_poses(run)
        self.assertAlmostEqual(report['render_zero_deg'], math.degrees(-0.4), places=5)
        self.assertAlmostEqual(report['total_rotation_deg'], 360 + math.degrees(0.1), places=4)
        point = np.array([0.5, 0.6, -0.3, 1.])
        for row in report['views'][::12]:
            obj, w2c, c2w = map(np.array, [row['object_transform'], row['world_to_camera'], row['camera_to_world']])
            np.testing.assert_allclose(w2c @ point, obj @ point + [0, 0, 5, 0], atol=1e-14)
            np.testing.assert_allclose(w2c @ c2w, np.eye(4), atol=1e-14)
            camera = np.r_[row['effective_camera_center'], 1.]
            np.testing.assert_allclose(w2c @ camera, [0, 0, 0, 1], atol=1e-14)
        np.testing.assert_allclose(report['views'][0]['relative_object_transform'], np.eye(4), atol=1e-14)

    def test_shuffled_partial_aruco_table_joins_source_filenames(self):
        run, dataset, _, _ = self.make_run()
        self.write_aruco(dataset, [
            {'frame_index': 72, 'filename': 'source_360.png', 'nominal_angle_deg': 360,
             'estimated_angle_deg': 359.3, 'angle_uncertainty_deg': 0.01, 'confidence': 'good'},
            {'frame_index': 0, 'filename': 'source_0.png', 'nominal_angle_deg': 0,
             'estimated_angle_deg': 0.2, 'angle_uncertainty_deg': 0, 'confidence': 'good'},
            {'frame_index': 1, 'filename': 'source_5.png', 'nominal_angle_deg': 5,
             'estimated_angle_deg': '', 'angle_uncertainty_deg': '', 'confidence': 'unresolved'},
        ])
        report = extract_view_poses(run)
        self.assertAlmostEqual(report['aruco_total_deg'], 359.1)
        self.assertIsNone(report['views'][1]['aruco_angle_deg'])
        self.assertIsNone(report['views'][2]['aruco_angle_deg'])
        self.assertEqual(report['views'][-1]['aruco_measured_angle_deg'], 359.3)

    def test_tae_uses_measured_time_and_bounded_anchored_corrections(self):
        run, dataset, predictor, _ = self.make_run(leverage_TAE=True,
                                                   max_residual_angle_deg=0.05, max_sweep_error_deg=0.15)
        measured = [0.2 + i * 359.1 / 72 + 0.02 * math.sin(math.pi * i / 72) for i in range(73)]
        folder = self.write_aruco(dataset, [
            {'frame_index': i, 'filename': f'source_{i * 5}.png', 'nominal_angle_deg': i * 5,
             'estimated_angle_deg': angle, 'angle_uncertainty_deg': 0.01, 'confidence': 'good'}
            for i, angle in enumerate(measured)
        ])
        (folder / 'summary.json').write_text(json.dumps({
            'input': {'image_count': 73},
            'geometry': {'rotation_centre_camera_mm': [0., 80., 570.],
                         'world_frame': {'basis_vectors_in_opencv_camera_coordinates': {'z': [0., 0.866, 0.5]}}},
        }))
        predictor.residuals.data.copy_(torch.linspace(-0.002, 0.003, 9).unsqueeze(0))
        predictor.sweep_error.data.fill_(0.001)
        predictor.save_weights(str(run), 30000)
        report = extract_view_poses(run)
        row = report['views'][23]
        time = (measured[23] - measured[0]) / (measured[-1] - measured[0])
        self.assertAlmostEqual(row['time'], time, places=7)
        self.assertNotAlmostEqual(row['time'], 23 / 72, places=6)
        expected = torch.tensor(math.radians(measured[23] - measured[0]), dtype=torch.float32)
        expected += predictor.angle_correction(torch.tensor(time, dtype=torch.float32), 0, True).detach()
        self.assertAlmostEqual(row['rotgs_render_angle_deg'], math.degrees(float(expected)), places=6)
        self.assertEqual(report['views'][0]['local_correction_deg'], 0)
        self.assertEqual(report['views'][-1]['local_correction_deg'], 0)
        self.assertLessEqual(max(abs(row['local_correction_deg']) for row in report['views']), 0.050001)

    def test_matching_pre_validation_residual_is_selected(self):
        run, _, predictor, cfg = self.make_run(checkpoint_iterations=[10000], test_iterations=[10000])
        snapshot = run / 'point_cloud/iteration_10000'
        snapshot.mkdir()
        for key in ['axis', 'center']:
            np.save(snapshot / f'{key}.npy', np.load(run / f'point_cloud/iteration_30000/{key}.npy'))
        predictor.residuals.data.fill_(0.1)
        predictor.save_weights(str(run), 9999)
        predictor.residuals.data.fill_(0.2)
        predictor.save_weights(str(run), 10000)
        report = extract_view_poses(run, 10000)
        self.assertIn('iteration_9999', report['residual_checkpoint'])
        self.assertAlmostEqual(report['render_zero_deg'], math.degrees(0.1), places=5)
        self.assertEqual(report['notes'], [])

    def test_missing_required_corrections_fail_instead_of_assuming_zero(self):
        run, _, _, _ = self.make_run()
        (run / 'residual_predictor/iteration_30000/residual_predictor.pth').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'missing learned angle corrections'):
            extract_view_poses(run)

    def test_no_correction_run_does_not_require_predictor_file(self):
        run, _, _, _ = self.make_run(wo_tiny=True, max_sweep_error_deg=0)
        (run / 'residual_predictor/iteration_30000/residual_predictor.pth').unlink()
        report = extract_view_poses(run)
        self.assertIsNone(report['residual_checkpoint'])
        self.assertAlmostEqual(report['total_rotation_deg'], 360, places=4)

    def test_noise_reproduction_preserves_callers_rng_state(self):
        run, _, _, _ = self.make_run(wo_tiny=True, max_sweep_error_deg=0, angle_noise_std=0.01)
        before = torch.random.get_rng_state().clone()
        first = extract_view_poses(run)
        # Predictor initialization uses zeros/linspace, so it consumes no RNG.
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = extract_view_poses(run)
        self.assertEqual(first['total_rotation_deg'], second['total_rotation_deg'])
        self.assertGreater(abs(first['total_rotation_deg'] - 360), 0.001)


if __name__ == '__main__':
    unittest.main()
