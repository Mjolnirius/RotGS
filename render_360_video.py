"""Render a smooth 360-degree turntable video from a trained RotGS model."""

from __future__ import annotations

import math
from argparse import ArgumentParser
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render, set_rasterizer
from scene import Scene
from scene.residual_predictor import ResidualPredictor
from utils.general_utils import safe_state


def render_360_video(
    dataset: ModelParams,
    pipeline: PipelineParams,
    *,
    iteration: int,
    frame_count: int,
    fps: int,
    output_path: Path | None,
    args,
) -> Path:
    """Render one full learned turntable rotation and encode it as MP4."""
    if frame_count < 2:
        raise ValueError("frame count must be at least 2")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not args.fixed_camera or args.multi_camera:
        raise ValueError("360 video rendering currently requires one fixed camera")

    gaussians = GaussianModel(
        dataset.sh_degree,
        fixed_camera=True,
        multi_camera=False,
        number_of_cameras=1,
        axis_mode=getattr(args, "axis_mode", "free"),
        axis_tilt_init_deg=getattr(args, "axis_tilt_init_deg", 30.0),
        axis_tilt_min_deg=getattr(args, "axis_tilt_min_deg", 0.0),
        axis_tilt_max_deg=getattr(args, "axis_tilt_max_deg", 90.0),
        axis_side_limit_deg=getattr(args, "axis_side_limit_deg", 5.0),
        center_max_offset=getattr(args, "center_max_offset", 0.25),
        center_warmup_iterations=getattr(args, "center_warmup_iterations", 2000),
    )
    scene = Scene(
        dataset,
        gaussians,
        fixed_camera=True,
        load_iteration=iteration,
        shuffle=False,
        random_init=True,
        multi_camera=False,
        eval_mode=True,
    )
    residual_predictor = ResidualPredictor(
        1,
        max_residual_angle_deg=getattr(args, "max_residual_angle_deg", 0.0),
        max_sweep_error_deg=getattr(args, "max_sweep_error_deg", 0.0),
    )
    residual_predictor.load_weights(
        dataset.model_path,
        iteration=scene.loaded_iter,
    )

    views = scene.getTrainCameras()
    if not views:
        raise ValueError("the trained scene has no camera view")
    view = views[0]
    background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(
        background_color,
        dtype=torch.float32,
        device="cuda",
    )
    rasterizer = set_rasterizer(view, gaussians, pipeline, background)

    if output_path is None:
        output_path = (
            Path(dataset.model_path)
            / "video"
            / f"ours_{scene.loaded_iter}"
            / "turntable_360.mp4"
        )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_size = (view.image_width, view.image_height)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 encoder for: {output_path}")

    direction = getattr(args, "rotation_direction", 1)
    if direction not in {-1, 1}:
        writer.release()
        raise ValueError("rotation direction must be -1 or 1")
    use_local_residual = not getattr(args, "wo_tiny", False)
    axis = gaussians.get_axis(0)
    center = gaussians.get_center(0)

    try:
        with torch.no_grad():
            for frame_index in tqdm(range(frame_count), desc="Rendering 360 video"):
                fraction = frame_index / frame_count
                coarse_angle = torch.tensor(
                    [direction * 2.0 * math.pi * fraction],
                    dtype=torch.float32,
                    device="cuda",
                )
                time = torch.tensor(fraction, dtype=torch.float32, device="cuda")
                angle = coarse_angle + residual_predictor.angle_correction(
                    time,
                    0,
                    use_local_residual=use_local_residual,
                )
                image = render(
                    rasterizer,
                    gaussians,
                    axis=axis,
                    center=center,
                    angle=angle,
                )["render"]
                rgb = (
                    image.clamp(0, 1)
                    .mul(255)
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"MP4 encoder did not produce a video: {output_path}")
    return output_path


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--frames", default=240, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fixed_camera", action="store_true", default=True)
    parser.add_argument("--wo_tiny", action="store_true", default=None)
    parser.add_argument("--max_residual_angle_deg", type=float, default=None)
    parser.add_argument("--max_sweep_error_deg", type=float, default=None)
    parser.add_argument(
        "--axis_mode",
        choices=("free", "bounded_tilt"),
        default=None,
    )
    parser.add_argument("--axis_tilt_init_deg", type=float, default=None)
    parser.add_argument("--axis_tilt_min_deg", type=float, default=None)
    parser.add_argument("--axis_tilt_max_deg", type=float, default=None)
    parser.add_argument("--axis_side_limit_deg", type=float, default=None)
    parser.add_argument("--center_max_offset", type=float, default=None)
    parser.add_argument("--center_warmup_iterations", type=int, default=None)
    parser.add_argument("--multi_camera", action="store_true", default=False)
    parser.add_argument("--name", type=str, default=None)
    args = get_combined_args(parser)

    safe_state(args.quiet)
    path = render_360_video(
        model.extract(args),
        pipeline.extract(args),
        iteration=args.iteration,
        frame_count=args.frames,
        fps=args.fps,
        output_path=getattr(args, "output", None),
        args=args,
    )
    print(f"360 video written to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
