"""Foreground-aware reconstruction metrics for saved RotGS test renders.

This module is deliberately independent of training.  RGB PNGs are evaluated in
their saved [0, 1] representation and the original source PNG alpha channel is
used as the soft foreground mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.loss_utils import ssim  # noqa: E402


@dataclass(frozen=True)
class Crop:
    raw_x0: int
    raw_y0: int
    raw_x1: int
    raw_y1: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        if image.mode != "RGBA":
            raise ValueError(f"source image has no RGBA alpha channel: {path}")
        rgba = np.asarray(image, dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    return rgb, alpha


def foreground_crop(alpha: np.ndarray, threshold: float = 0.001, padding: int = 8) -> Crop:
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be HxW, got {alpha.shape}")
    if padding < 0:
        raise ValueError("padding must be nonnegative")
    points = np.argwhere(alpha > threshold)
    if not len(points):
        raise ValueError("foreground mask is empty")
    raw_y0, raw_x0 = points.min(axis=0)
    raw_y1, raw_x1 = points.max(axis=0) + 1
    height, width = alpha.shape
    return Crop(
        int(raw_x0), int(raw_y0), int(raw_x1), int(raw_y1),
        max(0, int(raw_x0) - padding),
        max(0, int(raw_y0) - padding),
        min(width, int(raw_x1) + padding),
        min(height, int(raw_y1) + padding),
    )


def foreground_psnr(render: np.ndarray, gt: np.ndarray, alpha: np.ndarray) -> float:
    _validate_shapes(render, gt, alpha)
    denominator = 3.0 * float(alpha.sum(dtype=np.float64))
    if denominator <= 0.0:
        raise ValueError("foreground mask is empty")
    error = render.astype(np.float64) - gt.astype(np.float64)
    mse = float((error * error * alpha[..., None]).sum(dtype=np.float64) / denominator)
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def cropped_ssim(render: np.ndarray, gt: np.ndarray, crop: Crop, device: torch.device) -> float:
    render_tensor = _crop_tensor(render, crop, device)
    gt_tensor = _crop_tensor(gt, crop, device)
    with torch.inference_mode():
        return float(ssim(render_tensor, gt_tensor).item())


class CroppedLPIPS:
    """VGG LPIPS v0.1 evaluated per crop with canonical [-1, 1] inputs."""

    def __init__(self, device: torch.device):
        from lpipsPyTorch.modules.lpips import LPIPS

        self.device = device
        self.model = LPIPS(net_type="vgg", version="0.1").to(device).eval()

    def __call__(self, render: np.ndarray, gt: np.ndarray, crop: Crop) -> float:
        render_tensor = _crop_tensor(render, crop, self.device).mul(2.0).sub(1.0)
        gt_tensor = _crop_tensor(gt, crop, self.device).mul(2.0).sub(1.0)
        with torch.inference_mode():
            return float(self.model(render_tensor, gt_tensor).item())


def _crop_tensor(image: np.ndarray, crop: Crop, device: torch.device) -> torch.Tensor:
    array = np.ascontiguousarray(image[crop.y0:crop.y1, crop.x0:crop.x1])
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device)


def _validate_shapes(render: np.ndarray, gt: np.ndarray, alpha: np.ndarray) -> None:
    if render.shape != gt.shape or render.ndim != 3 or render.shape[2] != 3:
        raise ValueError(f"RGB shape mismatch: render={render.shape}, gt={gt.shape}")
    if alpha.shape != render.shape[:2]:
        raise ValueError(f"alpha shape mismatch: alpha={alpha.shape}, RGB={render.shape}")
