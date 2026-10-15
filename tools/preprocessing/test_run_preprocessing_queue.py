"""Queue checks using small fixtures and mocked GPU/process operations."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from tools.preprocessing import run_preprocessing_queue as queue


class PreprocessingQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exports = self.root / 'exports'
        self.exports.mkdir()
        self.source = self.exports / '4_Chips_Blau_30d'
        self.source.mkdir()
        self.outputs = self.root / 'cleaned'
        self.outputs.mkdir()
        self.output = self.outputs / '4_Chips_Blau_30d_SAM31AC_TAE'
        self.output.mkdir()
        self.logs = self.root / 'logs'
        cameras = self.root / 'cameras.txt'
        cameras.write_text('1 PINHOLE 16 16 10 10 8 8\n')
        calibration = self.root / 'calibration.npz'
        calibration.write_bytes(b'test calibration')
        checkpoint = self.root / 'checkpoint.pt'
        checkpoint.write_bytes(b'test checkpoint')
        self.recipe = dict(cameras=cameras, calibration=calibration, checkpoint=checkpoint,
                           cameras_sha256=queue.sha256(cameras),
                           calibration_sha256=queue.sha256(calibration),
                           crop_margin=10, segmentation_max_width=1600,
                           mask_threshold=0.5, marker_size_mm=40.0)
        self.prompt = 'potato chips bag'
        rows = [dict(source=f'frame{index}.png', output=f'{index:03d}.png', angle_degrees=index * 5)
                for index in range(2)]
        images = self.output / 'images'
        images.mkdir()
        for row in rows:
            with Image.new('RGBA', (16, 16), (10, 20, 30, 255)) as image:
                image.save(self.source / row['source'])
                image.save(images / row['output'])
        self.metadata = dict(
            input_folder=str(self.source), input_cameras_file=str(cameras),
            input_cameras_sha256=self.recipe['cameras_sha256'],
            alpha_mode='sam31_product', overwrite_alpha_mask=True,
            crop_selection_method='sam31_automatic', crop_mode='smallest_square',
            automatic_crop=dict(margin_pixels=10, working_size=[1600, 1067]),
            image_count=2, renamed_images=rows, output_size=[16, 16], crop_size=[16, 16],
            undistortion=dict(applied=True, calibration_file=str(calibration),
                              calibration_sha256=self.recipe['calibration_sha256']),
            segmentation=dict(backend='sam3.1', model='multiplex', checkpoint=str(checkpoint),
                              text_prompt=self.prompt, threshold_kind='probability', mask_threshold=0.5,
                              mode='per_frame_concept', input_view='cropped', inference_passes=2,
                              reused_crop_detection_masks=False, source_alpha_used=False,
                              inference_batch_size=1, postprocess_batch_size=1, working_size=[16, 16]),
            turntable_angle_estimation=dict(status='completed', marker_size_mm=40.0))
        self.baseline = self.root / 'baseline.json'
        self.baseline.write_text(json.dumps(self.metadata))
        self.write_metadata()
        for name in ('sam31_prompts.json', 'mask_review/index.html', 'crop_review/index.html',
                     'crop_review/detections.json', 'sparse/0/cameras.txt'):
            target = self.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(cameras.read_text() if name.endswith('cameras.txt') else '{}')
        tae = self.output / 'turntable_angle_estimation'
        tae.mkdir()
        (tae / 'summary.json').write_text(json.dumps(dict(
            input=dict(image_count=2), geometry=dict(
                world_frame=dict(basis_vectors_in_opencv_camera_coordinates=dict(z=[0, 1, 1])),
                rotation_centre_camera_mm=[0, 0, 100]))))
        (tae / 'turntable_angles.csv').write_text(
            'frame_index,filename,nominal_angle_deg,estimated_angle_deg,angle_uncertainty_deg,confidence\n'
            '0,frame0.png,0,0,0.01,high\n1,frame1.png,5,5.1,0.01,high\n')

    def write_metadata(self):
        (self.output / 'preprocessing_metadata.json').write_text(json.dumps(self.metadata))

    def problem(self):
        return queue.dataset_problem(self.output, self.source, self.recipe, self.prompt)

    def arguments(self):
        return ['--source-root', str(self.exports), '--output-root', str(self.outputs),
                '--baseline', str(self.baseline), '--log-root', str(self.logs)]

    def test_complete_dataset_is_reused_and_recipe_matches_baseline(self):
        self.assertEqual(queue.load_recipe(self.baseline), self.recipe)
        self.assertIsNone(self.problem())
        self.assertEqual(queue.completed_dataset(self.source, self.outputs, self.recipe, self.prompt), self.output)

    def test_old_masks_and_manual_crops_are_not_reused(self):
        self.metadata['segmentation']['inference_passes'] = 1
        self.write_metadata()
        self.assertIsNotNone(self.problem())
        self.metadata['segmentation']['inference_passes'] = 2
        self.metadata['crop_selection_method'] = 'provided'
        self.write_metadata()
        self.assertIsNotNone(self.problem())

    def test_missing_corrupt_and_rgb_images_are_not_complete(self):
        image_path = self.output / 'images/001.png'
        original = image_path.read_bytes()
        image_path.unlink()
        self.assertIsNotNone(self.problem())
        image_path.write_bytes(b'incomplete PNG')
        self.assertIsNotNone(self.problem())
        with Image.new('RGB', (16, 16)) as image:
            image.save(image_path)
        self.assertIsNotNone(self.problem())
        image_path.write_bytes(original)
        self.assertIsNone(self.problem())

    def test_tae_rows_must_match_all_images(self):
        table = self.output / 'turntable_angle_estimation/turntable_angles.csv'
        table.write_text('\n'.join(table.read_text().splitlines()[:2]) + '\n')
        self.assertIsNotNone(self.problem())

    def test_changed_calibration_is_not_silently_reused(self):
        self.recipe['calibration'].write_bytes(b'changed calibration')
        with self.assertRaisesRegex(ValueError, 'calibration has changed'):
            queue.load_recipe(self.baseline)
        self.metadata['undistortion']['calibration_sha256'] = 'other calibration'
        self.write_metadata()
        self.assertIsNotNone(self.problem())

    def test_archived_dataset_is_excluded_even_when_it_is_complete(self):
        archive = self.outputs / 'Outdated_Preprocessing_Files'
        archive.mkdir()
        self.output.rename(archive / self.output.name)
        self.assertIsNone(queue.completed_dataset(self.source, self.outputs, self.recipe, self.prompt))

    def test_dry_run_filters_views_and_does_not_write_or_start_jobs(self):
        shutil.rmtree(self.output)
        (self.exports / '4_Chips_Blau_5d').mkdir()
        (self.exports / '4_Chips_Blau_60d').mkdir()
        (self.exports / '4_Chips_Blau_30d copy').mkdir()
        output = io.StringIO()
        with patch.object(queue, 'run_job') as run, patch.object(queue.fcntl, 'flock') as flock:
            with redirect_stdout(output):
                self.assertEqual(queue.main([*self.arguments(), '--dry-run']), 0)
            run.assert_not_called()
            flock.assert_not_called()
        self.assertFalse(self.logs.exists())
        self.assertIn('1 sequences: 0 already complete, 1 pending.', output.getvalue())
        command = queue.preprocessing_command(self.source, self.output, self.recipe, self.prompt)
        self.assertEqual(command[command.index('--text-prompt') + 1], self.prompt)
        for option in ('--width', '--skip-tae', '--interactive', '--selection', '--manual_crop', '--box-only'):
            self.assertNotIn(option, command)

    def test_gpu_idle_streak_resets_on_foreign_allocation(self):
        samples = [(0, [], 'idle')] * 2 + [(0, ['foreign'], 'busy')] + [(0, [], 'idle')] * 5
        with patch.object(queue, 'gpu_snapshot', side_effect=samples) as snapshot:
            with patch.object(queue, 'record_gpu'), patch.object(queue.time, 'sleep'), patch.object(queue, 'log'):
                queue.wait_for_gpu(0, self.logs)
        self.assertEqual(snapshot.call_count, 8)

    def test_gpu_snapshot_ignores_only_own_group_and_zero_allocations(self):
        with patch.object(queue.subprocess, 'check_output', side_effect=[
                '0, 900, 32000', '111, python, 500\n222, python, 400\n333, python, 0']):
            with patch.object(queue.os, 'getpgid', side_effect=[999, 222]):
                utilization, foreign, _ = queue.gpu_snapshot(0, own_pgid=999)
        self.assertEqual(utilization, 0)
        self.assertEqual(foreign, ['222:python:400MiB'])

    def test_failed_jobs_retry_without_publishing_or_overwriting(self):
        shutil.rmtree(self.output)
        with patch.object(queue, 'ROOT', self.root), patch.object(queue, 'run_job', return_value=1) as run:
            with patch.object(queue, 'wait_for_gpu'), patch.object(queue, 'wait_for_disk_space'):
                with patch.object(queue.time, 'sleep'), patch.object(queue, 'log'), redirect_stdout(io.StringIO()):
                    with patch.object(queue.shutil, 'which', return_value='/usr/bin/nvidia-smi'):
                        with patch.object(queue.importlib.util, 'find_spec', return_value=object()):
                            with patch.dict(queue.os.environ):
                                self.assertEqual(queue.main(self.arguments()), 1)
        self.assertEqual(run.call_count, 3)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
