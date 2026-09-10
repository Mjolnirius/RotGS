"""Focused tests for loss helpers used by multi-camera calibration."""

import unittest

import torch

from utils.loss_utils import silhouette_iou_loss


class SilhouetteIouLossTests(unittest.TestCase):
    def test_identical_masks_have_zero_loss(self) -> None:
        mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        self.assertAlmostEqual(silhouette_iou_loss(mask, mask).item(), 0.0)

    def test_excess_area_is_penalized(self) -> None:
        target = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        oversized = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

        self.assertAlmostEqual(
            silhouette_iou_loss(oversized, target).item(), 0.5
        )

    def test_loss_provides_gradient_outside_target(self) -> None:
        target = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        predicted = torch.tensor(
            [[0.5, 0.8], [0.0, 0.0]], requires_grad=True
        )

        loss = silhouette_iou_loss(predicted, target)
        loss.backward()

        self.assertGreater(predicted.grad[0, 0].item(), 0.0)
        self.assertLess(predicted.grad[0, 1].item(), 0.0)

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            silhouette_iou_loss(torch.zeros(2, 2), torch.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
