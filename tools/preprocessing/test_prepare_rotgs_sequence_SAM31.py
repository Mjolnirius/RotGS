"""Dataset safety/geometry checks; real SAM inference is a separate smoke test."""
import json
import inspect
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from tools.preprocessing import prepare_rotgs_sequence_SAM31 as sam
from contextlib import nullcontext


class FakeSegmenter:
    def __init__(self, frames_folder, *args):
        with Image.open(frames_folder / "0000.jpg") as image:
            self.size = image.size

    def close(self):
        pass

    def detect(self, index, box=None, *, preserve_components=False):
        self.last_detection_method = 'text'
        mask = np.zeros((self.size[1], self.size[0]), dtype=bool)
        mask[4:20, 8:16] = True
        return mask

    def track(self, prompt, end, reverse=False):
        anchor = prompt["frame_index"]
        for index in range(anchor, end + (-1 if reverse else 1), -1 if reverse else 1):
            mask = np.zeros((self.size[1], self.size[0]), dtype=bool)
            mask[4:20, 8:16] = True
            yield index, mask


class SAM31Tests(unittest.TestCase):
    def adapter(self):
        class Predictor:
            def __init__(self):
                self.requests = []
                self.stream_requests = []
                self.stream_closed = False
                self.baseline_frames = set()
                self.refined = False

            def output(self, frame_index):
                masks = np.zeros((2, 30, 40), dtype=bool)
                masks[0, 1:29, 24:39] = True  # unrelated object
                masks[1, 4:25, 4:20] = True  # product
                return dict(frame_index=frame_index, outputs=dict(
                    out_obj_ids=np.array([7, 42]), out_binary_masks=masks))

            def handle_request(self, request):
                self.requests.append(request)
                if request['type'] == 'add_prompt':
                    if 'points' in request:
                        self.refined = True
                    return self.output(request['frame_index'])
                if request['type'] == 'reset_session':
                    self.baseline_frames.clear()
                    self.refined = False
                return dict(is_success=True)

            def propagate_in_video(self, **kwargs):
                self.assert_last_batch = kwargs['is_last_batch']
                request = dict(kwargs, start_frame_index=kwargs['start_frame_idx'])
                self.stream_requests.append(request)
                step = -1 if request['propagation_direction'] == 'backward' else 1
                try:
                    for offset in range(request['max_frame_num_to_track'] + 1):
                        index = request['start_frame_index'] + step * offset
                        if self.refined and index not in self.baseline_frames:
                            raise AssertionError('point propagation used an uncached frame')
                        self.baseline_frames.add(index)
                        yield self.output(index)
                finally:
                    self.stream_closed = True

            def shutdown(self):
                self.requests.append(dict(type='shutdown'))

        adapter = sam.SAM31Segmenter.__new__(sam.SAM31Segmenter)
        adapter.torch = SimpleNamespace(inference_mode=nullcontext,
                                        cuda=SimpleNamespace(empty_cache=Mock()))
        adapter.predictor = Predictor()
        adapter.session_id = 'test'
        adapter.object_id = None
        adapter.size = (40, 30)
        adapter.threshold = 0.5
        adapter.text_prompt = 'beverage can'
        return adapter

    def prompt(self):
        return dict(frame_index=3, box=[4, 3, 20, 27],
                    points=[[10, 10], [32, 10]], labels=[1, 0])

    def test_box_conversion_instance_selection_and_separate_point_refinement(self):
        adapter = self.adapter()
        mask = adapter.preview(self.prompt())
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[10, 32])
        requests = adapter.predictor.requests
        self.assertEqual(requests[0]['type'], 'reset_session')
        np.testing.assert_allclose(requests[1]['bounding_boxes'], [[.1, .1, .4, .8]])
        self.assertEqual(requests[1]['text'], 'beverage can')
        self.assertNotIn('points', requests[1])
        self.assertEqual(requests[2]['type'], 'remove_object')
        self.assertEqual(requests[2]['obj_id'], 7)
        self.assertEqual(requests[3]['obj_id'], 42)
        self.assertFalse(requests[3]['rel_coordinates'])
        self.assertNotIn('text', requests[3])
        self.assertNotIn('bounding_boxes', requests[3])

    def test_anchor_clicks_apply_only_to_anchor_and_each_view_resets(self):
        adapter = self.adapter()
        self.assertEqual([i for i, _ in adapter.track(self.prompt(), 5)], [3, 4, 5])
        self.assertEqual([i for i, _ in adapter.track(self.prompt(), 1, True)], [3, 2, 1])
        self.assertEqual(sum(r['type'] == 'reset_session' for r in adapter.predictor.requests), 6)
        self.assertEqual(adapter.predictor.stream_requests, [])
        point_requests = [r for r in adapter.predictor.requests if 'points' in r]
        self.assertEqual([r['frame_index'] for r in point_requests], [3, 3])
        predictor = adapter.predictor
        exited = []
        predictor.bf16_context = SimpleNamespace(
            __exit__=lambda *args: exited.append('predictor'))
        tracker = SimpleNamespace(bf16_context=SimpleNamespace(
            __exit__=lambda *args: exited.append('tracker')))
        predictor.model = SimpleNamespace(tracker=tracker)
        adapter.close()
        self.assertEqual(exited, ['predictor', 'tracker'])
        self.assertIsNone(predictor.bf16_context)
        self.assertIsNone(tracker.bf16_context)
        self.assertIsNone(adapter.session_id)
        self.assertIsNone(adapter.predictor)
        self.assertEqual([r['type'] for r in predictor.requests[-2:]],
                         ['close_session', 'shutdown'])
        adapter.torch.cuda.empty_cache.assert_called_once()
        adapter.close()  # cleanup is idempotent

    def test_model_uses_single_frame_batches_before_loading_session(self):
        model = SimpleNamespace(batched_grounding_batch_size=16, postprocess_batch_size=16,
                                init_state=lambda **kwargs: None)
        requests = []

        def request(payload):
            requests.append(payload)
            self.assertEqual(model.batched_grounding_batch_size, 1)
            self.assertEqual(model.postprocess_batch_size, 1)
            return dict(session_id='test')

        predictor = SimpleNamespace(model=model, handle_request=request,
                                    shutdown=Mock(), bf16_context=None)
        cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                               empty_cache=Mock())
        modules = dict(torch=SimpleNamespace(cuda=cuda), sam3=SimpleNamespace())
        modules['sam3.model_builder'] = SimpleNamespace(build_sam3_predictor=lambda **kwargs: predictor)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new('RGB', (40, 30)).save(folder / '0000.jpg')
            with patch.dict('sys.modules', modules):
                segmenter = sam.SAM31Segmenter(folder, '/unused', text_prompt='beverage can')
                self.assertEqual(requests[0]['type'], 'start_session')
                segmenter.close()
        self.assertIsNone(segmenter.predictor)
        predictor.shutdown.assert_called_once()
        cuda.empty_cache.assert_called_once()

    def test_cleanup_cuda_failure_preserves_original_inference_error(self):
        adapter = self.adapter()
        adapter.torch.cuda.empty_cache.side_effect = RuntimeError('cleanup CUDA failure')
        with self.assertRaisesRegex(ValueError, 'original inference failure') as caught:
            try:
                raise ValueError('original inference failure')
            finally:
                adapter.close()
        self.assertIsNone(adapter.predictor)
        if hasattr(caught.exception, '__notes__'):
            self.assertIn('cleanup CUDA failure', caught.exception.__notes__[0])

    def test_missing_product_is_rejected_instead_of_switching_objects(self):
        adapter = self.adapter()
        adapter.object_id = 99
        with self.assertRaisesRegex(ValueError, 'tracked product missing'):
            adapter._mask(adapter.predictor.output(4))

    def test_automatic_detection_sends_text_without_clicks_or_model_boxes(self):
        adapter = self.adapter()
        mask = adapter.detect(4, [4, 3, 20, 27])
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[10, 32])
        request = adapter.predictor.requests[-1]
        self.assertEqual(request['text'], 'beverage can')
        self.assertNotIn('points', request)
        self.assertNotIn('bounding_boxes', request)
        with tempfile.TemporaryDirectory() as directory:
            rows = sam.detect_masks(adapter, 3, [4, 3, 20, 27], Path(directory) / 'masks')
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r['min_overlap_iou'] is None for r in rows))

    def test_missing_text_detection_retries_box_and_records_fallback(self):
        adapter = self.adapter()
        original = adapter.predictor.handle_request

        def request(payload):
            if payload['type'] == 'add_prompt' and 'bounding_boxes' not in payload:
                return dict(frame_index=payload['frame_index'], outputs=dict(
                    out_obj_ids=np.zeros(0, dtype=int),
                    out_binary_masks=np.zeros((0, 30, 40), dtype=bool)))
            return original(payload)

        adapter.predictor.handle_request = request
        self.assertTrue(adapter.detect(14, [4, 3, 20, 27])[10, 10])
        self.assertEqual(adapter.last_detection_method, 'text_and_box')

    def test_background_mask_is_rejected_and_shorter_text_is_tried(self):
        adapter = self.adapter()
        original = adapter.predictor.handle_request

        def request(payload):
            if payload['type'] == 'add_prompt' and payload.get('text') == 'beverage can':
                return dict(frame_index=payload['frame_index'], outputs=dict(
                    out_obj_ids=np.array([99]),
                    out_binary_masks=np.ones((1, 30, 40), dtype=bool)))
            return original(payload)

        adapter.predictor.handle_request = request
        mask = adapter.detect(14, [4, 3, 20, 27])
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[0, 0])
        self.assertEqual(adapter.last_detection_method, 'text_fallback')
        self.assertEqual(adapter.last_detection_prompt, 'can')

    def test_crop_detection_keeps_detached_parts_and_never_uses_a_full_image_box(self):
        adapter = self.adapter()
        original = adapter.predictor.handle_request

        def request(payload):
            response = original(payload)
            if payload['type'] == 'add_prompt':
                mask = np.zeros((30, 40), dtype=bool)
                mask[4:20, 8:16] = True
                mask[24:26, 26:29] = True  # detached part of the same object
                response['outputs'] = dict(out_obj_ids=np.array([42]),
                                           out_binary_masks=mask[None])
            return response

        adapter.predictor.handle_request = request
        mask = adapter.detect(0, preserve_components=True)
        self.assertTrue(mask[24, 26])
        self.assertEqual(sam._mask_box_in_source(mask, (80, 60)), (16, 8, 58, 52))
        self.assertFalse(adapter.detect(0, [4, 3, 32, 28])[24, 26])

        def missing(payload):
            if payload['type'] == 'add_prompt':
                self.assertNotIn('bounding_boxes', payload)
                return dict(frame_index=payload['frame_index'], outputs=dict(
                    out_obj_ids=np.zeros(0, dtype=int),
                    out_binary_masks=np.zeros((0, 30, 40), dtype=bool)))
            return original(payload)

        adapter.predictor.handle_request = missing
        with self.assertRaisesRegex(ValueError, 'automatic crop detection failed'):
            adapter.detect(1, preserve_components=True)

    def test_crop_box_mapping_rounds_outward_on_both_axes(self):
        mask = np.zeros((7, 11), dtype=bool)
        mask[2:5, 3:8] = True
        self.assertEqual(sam._mask_box_in_source(mask, (101, 65)), (27, 18, 74, 47))
        with self.assertRaisesRegex(ValueError, 'empty'):
            sam._mask_box_in_source(np.zeros_like(mask), (101, 65))

    def test_final_detection_preserves_holes_and_detached_object_parts(self):
        adapter = self.adapter()
        original = adapter.predictor.handle_request
        mask = np.zeros((30, 40), dtype=bool)
        mask[4:20, 8:16] = True
        mask[8:10, 10:12] = False  # internal opening
        mask[24:26, 26:29] = True  # detached part of the selected object

        def request(payload):
            response = original(payload)
            if payload['type'] == 'add_prompt':
                response['outputs'] = dict(out_obj_ids=np.array([42]),
                                           out_binary_masks=mask[None])
            return response

        adapter.predictor.handle_request = request
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'masks'
            rows = sam.detect_masks(adapter, 1, [4, 3, 32, 28], folder)
            with Image.open(folder / '0000.png') as image:
                np.testing.assert_array_equal(image, mask.astype(np.uint8) * 255)
            self.assertEqual(rows[0]['detection_prompt'], 'beverage can')
            self.assertIsNone(rows[0]['min_overlap_iou'])

    def test_probability_threshold_and_cpu_are_rejected_before_loading(self):
        for threshold in (0, 1, float('nan')):
            with self.assertRaisesRegex(ValueError, 'threshold'):
                sam.SAM31Segmenter('/unused', '/unused', threshold=threshold)
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            sam.SAM31Segmenter('/unused', '/unused', device='cpu')

    def test_interactive_crop_receives_six_undistorted_comparisons(self):
        from tools.preprocessing.rotgs_web_selection import _display_comparison_sources

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            Image.new("RGB", (40, 32), (30, 60, 90)).save(path)
            ordered = [SimpleNamespace(angle=angle, path=path)
                       for angle in sam.legacy.CROP_REVIEW_ANGLES]
            previews = []

            def select(first, *args, **kwargs):
                comparisons = kwargs["comparison_images"]
                previews.extend(p for _, p in comparisons)
                views = _display_comparison_sources(comparisons, False, (1400, 900), first.size)
                self.assertEqual([view[0] for view in views], list(sam.legacy.CROP_REVIEW_ANGLES))
                return (4, 4, 28, 28), (4, 4, 28, 28), None, 24

            with Image.open(path) as first, \
                 patch.object(sam.legacy, "_remap_rgb", side_effect=lambda image, _: image.convert("RGB")), \
                 patch.object(sam.legacy, "choose_processing_settings_web", side_effect=select):
                result = sam.shared._choose_crop(first, ordered, None, 1600, None, 8765)
            self.assertEqual(result[-1], 24)
            self.assertFalse(any(p.exists() for p in previews))

    def test_cli_maps_arguments_to_function(self):
        signature = inspect.signature(sam.prepare_rotgs_sequence_SAM31)
        argv = ["sam31", "source", "--cameras", "cameras.txt", "--checkpoint", "weights.pt",
                "--calibration", "calibration.npz", "--box-only", "--selection", "1", "2", "20", "30"]
        with patch("sys.argv", argv), patch.object(sam, "prepare_rotgs_sequence_SAM31") as prepare:
            prepare.side_effect = lambda **kwargs: signature.bind(**kwargs)
            self.assertEqual(sam.main(), 0)
            self.assertEqual(prepare.call_args.kwargs["calibration_file"], Path("calibration.npz"))
            self.assertFalse(prepare.call_args.kwargs['manual_crop'])
            self.assertEqual(prepare.call_args.kwargs['crop_margin'], 10)
        argv += ['--manual_crop', '--crop-margin', '15', '--diagnostics-dir', 'logs', '--cuda-sync']
        with patch('sys.argv', argv), patch.object(sam, 'prepare_rotgs_sequence_SAM31') as prepare:
            prepare.side_effect = lambda **kwargs: signature.bind(**kwargs)
            self.assertEqual(sam.main(), 0)
            self.assertTrue(prepare.call_args.kwargs['manual_crop'])
            self.assertEqual(prepare.call_args.kwargs['crop_margin'], 15)
            self.assertEqual(prepare.call_args.kwargs['diagnostics_dir'], Path('logs'))
            self.assertTrue(prepare.call_args.kwargs['cuda_sync'])

    def small_sequence(self, root):
        source = root / 'source'
        source.mkdir()
        for angle in range(0, 360, 5):
            Image.new('RGB', (100, 80), (30, 60, 90)).save(source / f'product_{angle}.png')
        camera = root / 'cameras.txt'
        camera.write_text('1 PINHOLE 100 80 100 100 50 40\n')
        calibration = root / 'camera_calibration.npz'
        np.savez(calibration, camera_matrix=np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]]),
                 dist_coeffs=np.zeros(5), image_size=[100, 80])
        checkpoint = root / 'test.pt'
        checkpoint.touch()
        return source, camera, dict(checkpoint=checkpoint, calibration_file=calibration,
                                   text_prompt='beverage can', skip_tae=True)

    def test_default_detects_every_view_then_segments_native_crop_without_browser(self):
        instances = []
        self_test = self

        class CropSegmenter(FakeSegmenter):
            def __init__(self, *args):
                if instances:
                    self_test.assertTrue(instances[0].closed, 'crop model must close before final segmentation')
                super().__init__(*args)
                self.detected = []
                self.closed = False
                instances.append(self)

            def close(self):
                self.closed = True

            def detect(self, index, box=None, *, preserve_components=False):
                self.detected.append(index)
                if box is not None:
                    return super().detect(index, box)
                self.assert_preserved = preserve_components
                self.last_detection_method = 'text'
                mask = np.zeros((self.size[1], self.size[0]), dtype=bool)
                if index == 37:  # widest extent lies between the six review angles
                    mask[25:65, 30:70] = True
                else:
                    mask[30:50, 40:60] = True
                return mask

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, camera, options = self.small_sequence(root)
            with patch.object(sam, 'SAM31Segmenter', CropSegmenter), \
                 patch('importlib.util.find_spec', return_value=object()), \
                 patch.object(sam.shared, '_choose_crop', side_effect=AssertionError('unexpected crop editor')), \
                 patch.object(sam.shared, 'prompt_browser', side_effect=AssertionError('unexpected mask editor')):
                summary = sam.prepare_rotgs_sequence_SAM31(source, camera, **options)
            output = Path(summary['output_folder'])
            self.assertEqual(output.name, 'source_SAM31AC_60')
            self.assertEqual(summary['metadata']['crop_box_left_top_right_bottom'], [20, 15, 80, 75])
            self.assertEqual(summary['metadata']['product_selection_left_top_right_bottom'], [30, 25, 70, 65])
            self.assertEqual(summary['metadata']['crop_selection_method'], 'sam31_automatic')
            discovery = summary['metadata']['automatic_crop']
            self.assertEqual(discovery['margin_pixels'], 10)
            self.assertEqual(len(discovery['frames']), 72)
            unpadded = sam.legacy.crop_box_for_selection((30, 25, 70, 65), (100, 80), True)
            self.assertEqual(summary['metadata']['crop_box_left_top_right_bottom'],
                             [unpadded[0]-10, unpadded[1]-10, unpadded[2]+10, unpadded[3]+10])
            self.assertEqual(len(instances), 2)
            self.assertTrue(instances[0].assert_preserved)
            self.assertTrue(all(instance.closed for instance in instances))
            self.assertTrue(all(instance.detected == list(range(72)) for instance in instances))
            self.assertEqual([instance.size for instance in instances], [(100, 80), (60, 60)])
            self.assertFalse(summary['metadata']['segmentation']['reused_crop_detection_masks'])
            self.assertEqual(summary['metadata']['segmentation']['working_size'], [60, 60])
            self.assertEqual(summary['metadata']['segmentation']['input_view'], 'cropped')
            self.assertEqual(summary['metadata']['segmentation']['inference_passes'], 2)
            self.assertTrue((output / 'crop_review/index.html').is_file())
            self.assertEqual(json.loads((output / 'crop_review/detections.json').read_text()), discovery)
            with Image.open(output / 'images/0000.png') as image:
                self.assertEqual(image.size, (60, 60))
                np.testing.assert_array_equal(np.asarray(image)[10, 10, :3], [30, 60, 90])
                expected_alpha = np.zeros((60, 60), dtype=np.uint8)
                expected_alpha[4:20, 8:16] = 255  # final crop-space mask differs from discovery
                np.testing.assert_array_equal(image.getchannel('A'), expected_alpha)
            with Image.open(output / 'images/0037.png') as image:
                self.assertEqual(np.count_nonzero(np.asarray(image.getchannel('A'))), 16 * 8)
            fields = (output / 'sparse/0/cameras.txt').read_text().split()
            self.assertEqual(fields[:4], ['1', 'PINHOLE', '60', '60'])
            self.assertEqual([float(v) for v in fields[4:]], [100, 100, 30, 25])
            with Image.open(output / 'crop_review/0037.png') as preview:
                self.assertEqual(preview.getpixel((20, 15)), (0, 255, 0))
                self.assertEqual(preview.getpixel((30, 25)), (255, 0, 255))

    def test_second_pass_failure_closes_both_sessions_without_publishing(self):
        instances = []

        class BrokenFinalSegmentation(FakeSegmenter):
            def __init__(self, *args):
                super().__init__(*args)
                self.closed = False
                instances.append(self)

            def close(self):
                self.closed = True

            def detect(self, index, box=None, *, preserve_components=False):
                if box is not None:
                    if index == 1:
                        raise RuntimeError('second pass failure')
                    return super().detect(index, box, preserve_components=preserve_components)
                self.last_detection_method = 'text'
                mask = np.zeros((self.size[1], self.size[0]), dtype=bool)
                mask[30:50, 40:60] = True
                return mask

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, camera, options = self.small_sequence(root)
            with patch.object(sam, 'SAM31Segmenter', BrokenFinalSegmentation), \
                 patch('importlib.util.find_spec', return_value=object()):
                with self.assertRaisesRegex(RuntimeError, 'second pass failure'):
                    sam.prepare_rotgs_sequence_SAM31(
                        source, camera, destination_folder=root / 'failed', **options)
            self.assertEqual(len(instances), 2)
            self.assertTrue(all(instance.closed for instance in instances))
            self.assertFalse((root / 'failed').exists())
            self.assertEqual(len(list(root.glob('failed.prompts-*.json'))), 1)
            self.assertFalse(list(root.glob('.failed.*')))

    def test_manual_crop_opens_old_editor_and_skips_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, camera, options = self.small_sequence(root)
            with patch.object(sam, 'SAM31Segmenter', FakeSegmenter), \
                 patch('importlib.util.find_spec', return_value=object()), \
                 patch.object(sam, '_discover_crop', side_effect=AssertionError('unexpected automatic crop')), \
                 patch.object(sam.shared, '_choose_crop', return_value=(
                     (20, 15, 80, 75), (20, 15, 80, 75), None, 60)) as editor:
                summary = sam.prepare_rotgs_sequence_SAM31(source, camera, manual_crop=True, **options)
            editor.assert_called_once()
            self.assertEqual(Path(summary['output_folder']).name, 'source_und_rn_roi_sqr_SAM31_60')
            self.assertEqual(summary['metadata']['crop_selection_method'], 'manual_six_view')
            self.assertIsNone(summary['metadata']['automatic_crop'])

    def test_failed_or_unfittable_crop_closes_session_without_publishing(self):
        instances = []

        class BrokenCrop(FakeSegmenter):
            def __init__(self, *args):
                super().__init__(*args)
                self.closed = False
                instances.append(self)

            def close(self):
                self.closed = True

            def detect(self, index, box=None, *, preserve_components=False):
                if index == 1:
                    raise ValueError('no product detected')
                return super().detect(index, box)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, camera, options = self.small_sequence(root)
            for segmenter, message in ((BrokenCrop, 'frame 1'), (FakeSegmenter, 'margin does not fit')):
                with self.subTest(segmenter=segmenter), \
                     patch.object(sam, 'SAM31Segmenter', segmenter), \
                     patch('importlib.util.find_spec', return_value=object()):
                    with self.assertRaisesRegex(ValueError, message):
                        sam.prepare_rotgs_sequence_SAM31(
                            source, camera, destination_folder=root / 'failed', **options)
                self.assertFalse((root / 'failed').exists())
            self.assertTrue(all(instance.closed for instance in instances))
            with self.assertRaisesRegex(ValueError, '--text-prompt'):
                sam.prepare_rotgs_sequence_SAM31(source, camera, **dict(options, text_prompt=None))
            with self.assertRaisesRegex(ValueError, 'crop margin'):
                sam.prepare_rotgs_sequence_SAM31(source, camera, **dict(options, crop_margin=-1))

    def test_complete_export_preserves_rgb_replaces_alpha_and_updates_intrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            rgb = np.full((32, 40, 4), [30, 60, 90, 255], dtype=np.uint8)
            rgb[0:4, 0:4, :3] = 255  # marker, opaque in old alpha
            for angle in range(0, 360, 5):
                Image.fromarray(rgb).save(source / f"product_{angle}.png")
            camera = root / "cameras.txt"
            camera.write_text("1 PINHOLE 40 32 100 100 20 16\n")
            calibration = root / "camera_calibration.npz"
            np.savez(calibration, camera_matrix=np.array([[100, 0, 20], [0, 100, 16], [0, 0, 1]]),
                     dist_coeffs=np.zeros(5), image_size=[40, 32])
            checkpoint = root / "test.pt"
            checkpoint.touch()
            output = root / "output"
            options = dict(checkpoint=checkpoint, calibration_file=calibration,
                           selection=(4, 4, 28, 28), destination_folder=output, box_only=True)
            def estimate(input_folder, intrinsics, staging, marker_size):
                folder = staging / 'turntable_angle_estimation'
                folder.mkdir()
                (folder / 'test-artifact.txt').write_text('synthetic TAE result')

            with patch.object(sam, "SAM31Segmenter", FakeSegmenter), \
                 patch.object(sam.legacy, '_run_turntable_angle_estimation', side_effect=estimate) as tae, \
                 patch("importlib.util.find_spec", return_value=object()):
                summary = sam.prepare_rotgs_sequence_SAM31(source, camera, **options)
                self.assertEqual(summary["processed"], 72)
                tae.assert_called_once()
                self.assertEqual(tae.call_args.args[0], source)
                self.assertEqual(tae.call_args.args[1], calibration)
                self.assertEqual(tae.call_args.args[3], 40.0)
                self.assertEqual(summary['metadata']['turntable_angle_estimation']['status'], 'completed')
                self.assertTrue((output / 'turntable_angle_estimation/test-artifact.txt').exists())
                with self.assertRaises(FileExistsError):
                    sam.prepare_rotgs_sequence_SAM31(source, camera, **options)
            with Image.open(output / "images" / "0000.png") as result:
                values = np.asarray(result)
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (24, 24))
                self.assertEqual(values[0, 0, 3], 0)
                self.assertEqual(values[10, 10, 3], 255)
                np.testing.assert_array_equal(values[10, 10, :3], [30, 60, 90])
            with Image.open(source / "product_0.png") as original:
                np.testing.assert_array_equal(original, rgb)
            fields = (output / "sparse" / "0" / "cameras.txt").read_text().split()
            self.assertEqual(fields[:4], ["1", "PINHOLE", "24", "24"])
            self.assertEqual([float(v) for v in fields[4:]], [100, 100, 16, 12])
            self.assertTrue((output / "mask_review" / "index.html").exists())
            saved = json.loads((output / "sam31_prompts.json").read_text())
            self.assertEqual(saved["coordinate_size"], [24, 24])
            self.assertEqual(len(saved["images"]), 72)
            recovery_files = list(output.glob("output.prompts-*.json"))
            self.assertEqual(len(recovery_files), 1)
            self.assertEqual(json.loads(recovery_files[0].read_text()), saved)
            self.assertFalse(list(root.glob("output.prompts-*.json")))
            automatic_options = dict(options, box_only=False, text_prompt='beverage can',
                                     destination_folder=root / 'automatic', skip_tae=True)
            with patch.object(sam, 'SAM31Segmenter', FakeSegmenter), \
                 patch.object(sam.legacy, '_run_turntable_angle_estimation') as tae, \
                 patch('importlib.util.find_spec', return_value=object()), \
                 patch.object(sam.shared, 'prompt_browser', side_effect=AssertionError('unexpected editor')):
                automatic = sam.prepare_rotgs_sequence_SAM31(source, camera, **automatic_options)
                tae.assert_not_called()
            self.assertEqual(automatic['metadata']['turntable_angle_estimation']['status'], 'skipped')
            self.assertEqual(automatic['metadata']['segmentation']['mode'], 'per_frame_concept')
            self.assertEqual(automatic['processed'], 72)
            self.assertTrue(all(row['min_overlap_iou'] is None for row in automatic['metadata']['mask_review']))
            recovery_files = list((root / 'automatic').glob('automatic.prompts-*.json'))
            self.assertEqual(len(recovery_files), 1)
            self.assertEqual(json.loads(recovery_files[0].read_text()),
                             json.loads((root / 'automatic' / 'sam31_prompts.json').read_text()))
            self.assertFalse(list(root.glob('automatic.prompts-*.json')))
            class BrokenTracker(FakeSegmenter):
                def track(self, *args, **kwargs):
                    raise ValueError("frame 5: empty mask")

            options["destination_folder"] = root / "failed"
            options['skip_tae'] = True
            with patch.object(sam, "SAM31Segmenter", BrokenTracker), \
                 patch("importlib.util.find_spec", return_value=object()):
                with self.assertRaisesRegex(ValueError, "frame 5"):
                    sam.prepare_rotgs_sequence_SAM31(source, camera, **options)
            self.assertFalse((root / "failed").exists())
            self.assertEqual(len(list(root.glob("failed.prompts-*.json"))), 1)
            self.assertFalse(list(root.glob(".failed.*")))
