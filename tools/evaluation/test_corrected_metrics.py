from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrected_metrics import foreground_crop, foreground_psnr, cropped_ssim


class CorrectedMetricTests(unittest.TestCase):
    def test_soft_mask_psnr_uses_three_channels_and_soft_weights(self):
        gt = np.zeros((2, 2, 3), dtype=np.float32)
        render = np.zeros_like(gt)
        render[0, 0] = 1.0
        alpha = np.array([[0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
        expected_mse = 1.5 / 3.0
        expected = 20.0 * np.log10(1.0 / np.sqrt(expected_mse))
        self.assertAlmostEqual(foreground_psnr(render, gt, alpha), expected, places=10)

    def test_crop_threshold_padding_and_clipping(self):
        alpha = np.zeros((10, 12), dtype=np.float32)
        alpha[2:8, 3:11] = 0.5
        alpha[0, 0] = 0.001
        crop = foreground_crop(alpha, threshold=0.001, padding=4)
        self.assertEqual((crop.raw_x0, crop.raw_y0, crop.raw_x1, crop.raw_y1), (3, 2, 11, 8))
        self.assertEqual((crop.x0, crop.y0, crop.x1, crop.y1), (0, 0, 12, 10))

    def test_identical_crop_ssim_is_one(self):
        rng = np.random.default_rng(4)
        image = rng.random((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32), dtype=np.float32)
        crop = foreground_crop(alpha)
        value = cropped_ssim(image, image, crop, torch.device("cpu"))
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_empty_mask_fails(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            foreground_crop(np.zeros((8, 8), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
