"""Dataset safety/geometry checks; real SAM inference is a separate smoke test."""
import json
import inspect
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from tools.preprocessing import prepare_rotgs_sequence_SAM21 as sam


class FakeSegmenter:
    def __init__(self, frames_folder, *args):
        with Image.open(frames_folder / "0000.jpg") as image:
            self.size = image.size

    def track(self, prompt, end, reverse=False):
        anchor = prompt["frame_index"]
        for index in range(anchor, end + (-1 if reverse else 1), -1 if reverse else 1):
            mask = np.zeros((self.size[1], self.size[0]), dtype=bool)
            mask[4:20, 8:16] = True
            yield index, mask


class SAM21Tests(unittest.TestCase):
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
                result = sam._choose_crop(first, ordered, None, 1600, None, 8765)
            self.assertEqual(result[-1], 24)
            self.assertFalse(any(p.exists() for p in previews))

    def test_editor_preview_and_save_http_flow(self):
        ready = threading.Event()
        servers, outcomes = [], []

        class Server(sam.HTTPServer):
            def server_activate(self):
                super().server_activate()
                servers.append(self)
                ready.set()

        class Segmenter:
            def preview(self, prompt):
                mask = np.zeros((16, 16), dtype=bool)
                mask[4:12, 4:12] = True
                return mask

        doc = dict(version=1, coordinate_size=[16, 16], images=["product_0.png"],
                   frames=[dict(frame_index=0, box=[3, 3, 13, 13], points=[], labels=[])])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(sam, "HTTPServer", Server), \
             patch.object(sam.secrets, "token_urlsafe", return_value="test-token"):
            path = Path(directory)
            Image.new("RGB", (16, 16)).save(path / "0000.jpg")
            thread = threading.Thread(target=lambda: outcomes.append(sam.prompt_browser(
                Segmenter(), path, doc, 0)), daemon=True)
            thread.start()
            self.assertTrue(ready.wait(5))
            address = f"http://127.0.0.1:{servers[0].server_port}"
            try:
                with urlopen(address + "/", timeout=5) as response:
                    self.assertEqual(response.geturl(), address + "/?token=test-token")
                    self.assertIn("text/html", response.headers["Content-Type"])
                    self.assertIn(b"Select the product", response.read())
                with self.assertRaises(HTTPError) as denied:
                    urlopen(address + "/config", timeout=5)
                self.assertEqual(denied.exception.code, 403)
                with urlopen(address + "/config?token=test-token", timeout=5) as response:
                    self.assertEqual(json.load(response), doc)
                with urlopen(Request(address + "/preview", data=json.dumps(doc).encode(),
                                     headers={"X-SAM-Token": "test-token"}), timeout=5) as response:
                    self.assertEqual(response.headers["Content-Type"], "image/png")
                    self.assertTrue(response.read().startswith(b"\x89PNG"))
            finally:
                with urlopen(Request(address + "/finish", data=json.dumps(doc).encode(),
                                     headers={"X-SAM-Token": "test-token"}), timeout=5) as response:
                    self.assertEqual(response.status, 200)
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(outcomes[0], doc["frames"])

    def test_cli_maps_arguments_to_function(self):
        signature = inspect.signature(sam.prepare_rotgs_sequence_SAM21)
        argv = ["sam21", "source", "--cameras", "cameras.txt", "--checkpoint", "weights.pt",
                "--calibration", "calibration.npz", "--box-only", "--selection", "1", "2", "20", "30"]
        with patch("sys.argv", argv), patch.object(sam, "prepare_rotgs_sequence_SAM21") as prepare:
            prepare.side_effect = lambda **kwargs: signature.bind(**kwargs)
            self.assertEqual(sam.main(), 0)
            self.assertEqual(prepare.call_args.kwargs["calibration_file"], Path("calibration.npz"))

    def test_rejects_stale_coordinates_and_out_of_bounds_prompts(self):
        doc = dict(version=1, coordinate_size=[40, 30],
                   frames=[dict(frame_index=0, box=[1, 1, 20, 25], points=[], labels=[])])
        self.assertEqual(sam.validate_prompts(doc, 10, (40, 30))[0]["frame_index"], 0)
        with self.assertRaises(ValueError):
            sam.validate_prompts(doc, 10, (80, 60))
        doc["frames"][0].update(points=[[40, 3]], labels=[1])
        with self.assertRaises(ValueError):
            sam.validate_prompts(doc, 10, (40, 30))
        doc["frames"][0].update(points=[[3, 3]], labels=[0], box=None)
        with self.assertRaises(ValueError):
            sam.validate_prompts(doc, 10, (40, 30))

    def test_component_filter_preserves_holes_and_positive_seed(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[2:10, 2:10] = True  # product
        mask[5:7, 5:7] = False  # real internal opening
        mask[15:29, 15:29] = True  # larger false-positive region
        prompt = dict(points=[[3, 3]], labels=[1])
        result = sam.product_component(mask, prompt)
        self.assertTrue(result[3, 3])
        self.assertFalse(result[6, 6])
        self.assertFalse(result[20, 20])
        with self.assertRaises(ValueError):
            sam.product_component(np.zeros_like(mask))

    def test_bidirectional_nearest_anchor_merge_and_disagreement(self):
        class Tracker:
            def track(self, prompt, end, reverse=False):
                anchor = prompt["frame_index"]
                for i in range(anchor, end + (-1 if reverse else 1), -1 if reverse else 1):
                    mask = np.zeros((10, 10), dtype=bool)
                    mask[2:7, (2 if anchor == 2 else 4):8] = True
                    yield i, mask

        with tempfile.TemporaryDirectory() as directory:
            rows = sam.propagate_masks(Tracker(), [{"frame_index": 2}, {"frame_index": 6}],
                                       9, Path(directory) / "masks")
            self.assertEqual([r["anchor"] for r in rows], [2, 2, 2, 2, 2, 6, 6, 6, 6])
            self.assertLess(rows[4]["min_overlap_iou"], 0.9)
            self.assertEqual(len(list((Path(directory) / "masks").glob("*.png"))), 9)

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
            with patch.object(sam, "SAM21Segmenter", FakeSegmenter), \
                 patch("importlib.util.find_spec", return_value=object()):
                summary = sam.prepare_rotgs_sequence_SAM21(source, camera, **options)
                self.assertEqual(summary["processed"], 72)
                with self.assertRaises(FileExistsError):
                    sam.prepare_rotgs_sequence_SAM21(source, camera, **options)
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
            saved = json.loads((output / "sam21_prompts.json").read_text())
            self.assertEqual(saved["coordinate_size"], [24, 24])
            self.assertEqual(len(saved["images"]), 72)
            class BrokenTracker(FakeSegmenter):
                def track(self, *args, **kwargs):
                    raise ValueError("frame 5: empty mask")

            options["destination_folder"] = root / "failed"
            with patch.object(sam, "SAM21Segmenter", BrokenTracker), \
                 patch("importlib.util.find_spec", return_value=object()):
                with self.assertRaisesRegex(ValueError, "frame 5"):
                    sam.prepare_rotgs_sequence_SAM21(source, camera, **options)
            self.assertFalse((root / "failed").exists())
            self.assertEqual(len(list(root.glob("failed.prompts-*.json"))), 1)
            self.assertFalse(list(root.glob(".failed.*")))


if __name__ == "__main__":
    unittest.main()
