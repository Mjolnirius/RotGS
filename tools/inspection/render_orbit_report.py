"""Render RotGS orbit media and maintain an incremental HTML report."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import GaussianModel, render, set_rasterizer
from scene import Scene
from scene.cameras import MiniCam
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


TRAJECTORIES = (
    ("orbit_30deg", "Original-angle 360", "Product rotation from the trained camera elevation."),
    ("orbit_front", "Front / level 360", "Product rotation from a level, 0-degree view."),
    ("orbit_top", "Top-view 360", "Product rotation around its vertical axis from above."),
    ("orbit_vertical", "Vertical orbit", "Bottom to front to top to back to bottom."),
)


def load_cfg(model_path: Path) -> argparse.Namespace:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing RotGS configuration: {cfg_path}")
    cfg = eval(cfg_path.read_text(), {"Namespace": argparse.Namespace})  # noqa: S307
    cfg.model_path = str(model_path)
    return cfg


def latest_iteration(model_path: Path) -> int:
    candidates = []
    for path in (model_path / "point_cloud").glob("iteration_*"):
        try:
            iteration = int(path.name.removeprefix("iteration_"))
        except ValueError:
            continue
        if (path / "point_cloud.ply").is_file():
            candidates.append(iteration)
    if not candidates:
        raise FileNotFoundError(f"no point-cloud checkpoint found under {model_path}")
    return max(candidates)


def normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("cannot normalise a near-zero vector")
    return vector / norm


def virtual_camera(template, position, target, up_hint) -> MiniCam:
    forward = normalise(target - position)
    right = normalise(np.cross(forward, normalise(up_hint)))
    up = normalise(np.cross(right, forward))
    camera_to_world = np.column_stack((right, -up, forward))
    rotation = camera_to_world.astype(np.float32)
    translation = (-rotation.T @ position).astype(np.float32)
    world_view = torch.tensor(
        getWorld2View2(rotation, translation),
        dtype=torch.float32,
        device="cuda",
    ).transpose(0, 1)
    projection = getProjectionMatrix(
        znear=template.znear,
        zfar=template.zfar,
        fovX=template.FoVx,
        fovY=template.FoVy,
    ).transpose(0, 1).cuda()
    full_projection = world_view.unsqueeze(0).bmm(
        projection.unsqueeze(0)
    ).squeeze(0)
    return MiniCam(
        int(template.image_width),
        int(template.image_height),
        template.FoVy,
        template.FoVx,
        template.znear,
        template.zfar,
        world_view,
        full_projection,
    )


def camera_frame(model, template):
    axis = normalise(
        model.get_axis(0).detach().cpu().numpy().astype(np.float64)
    )
    center = (
        model.get_center(0).detach().cpu().numpy().astype(np.float64)
    )
    position = (
        template.camera_center.detach().cpu().numpy().astype(np.float64)
    )
    camera_vector = position - center
    distance = float(np.linalg.norm(camera_vector))

    # The learned axis sign is arbitrary. Canonical +z faces the elevated
    # training camera, which makes "top" deterministic for these captures.
    canonical_z = axis.copy()
    if float(np.dot(camera_vector, canonical_z)) < 0:
        canonical_z *= -1.0
    canonical_front = normalise(
        camera_vector - np.dot(camera_vector, canonical_z) * canonical_z
    )
    elevation = math.degrees(
        math.asin(
            np.clip(
                np.dot(normalise(camera_vector), canonical_z), -1.0, 1.0
            )
        )
    )
    return axis, center, distance, canonical_z, canonical_front, elevation


def trajectory_camera(
    name,
    fraction,
    template,
    center,
    distance,
    canonical_z,
    canonical_front,
):
    if name == "orbit_30deg":
        return template
    if name == "orbit_front":
        return virtual_camera(
            template,
            center + distance * canonical_front,
            center,
            canonical_z,
        )
    if name == "orbit_top":
        return virtual_camera(
            template,
            center + distance * canonical_z,
            center,
            canonical_front,
        )
    if name == "orbit_vertical":
        theta = -0.5 * math.pi + 2.0 * math.pi * fraction
        radial = (
            math.cos(theta) * canonical_front
            + math.sin(theta) * canonical_z
        )
        tangent = (
            -math.sin(theta) * canonical_front
            + math.cos(theta) * canonical_z
        )
        return virtual_camera(
            template, center + distance * radial, center, tangent
        )
    raise ValueError(f"unknown trajectory: {name}")


class MediaWriters:
    def __init__(
        self,
        mp4_path,
        gif_path,
        poster_path,
        *,
        fps,
        gif_fps,
        gif_width,
        source_size,
    ):
        self.poster_path = poster_path
        self.gif_stride = max(1, round(fps / gif_fps))
        self.gif_width = gif_width
        self.frame_index = 0
        width, height = source_size
        self.video_size = (width - width % 2, height - height % 2)
        self.mp4 = imageio.get_writer(
            str(mp4_path),
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=None,
            ffmpeg_log_level="error",
        )
        self.gif = imageio.get_writer(
            str(gif_path),
            format="GIF-PIL",
            mode="I",
            duration=1.0 / gif_fps,
            loop=0,
        )

    def append(self, rgb):
        if self.frame_index == 0:
            Image.fromarray(rgb).save(
                self.poster_path, quality=88, optimize=True
            )
        width, height = self.video_size
        self.mp4.append_data(rgb[:height, :width])
        if self.frame_index % self.gif_stride == 0:
            image = Image.fromarray(rgb)
            if image.width > self.gif_width:
                gif_height = round(
                    image.height * self.gif_width / image.width
                )
                image = image.resize(
                    (self.gif_width, gif_height), Image.Resampling.LANCZOS
                )
            self.gif.append_data(np.asarray(image))
        self.frame_index += 1

    def close(self):
        self.mp4.close()
        self.gif.close()


def render_trajectory(
    name,
    output_dir,
    *,
    gaussians,
    pipeline,
    template,
    axis,
    center_tensor,
    center,
    camera_distance,
    canonical_z,
    canonical_front,
    background,
    direction,
    frames,
    fps,
    gif_fps,
    gif_width,
):
    mp4_path = output_dir / f"{name}.mp4"
    gif_path = output_dir / f"{name}.gif"
    poster_path = output_dir / f"{name}_poster.jpg"
    writers = MediaWriters(
        mp4_path,
        gif_path,
        poster_path,
        fps=fps,
        gif_fps=gif_fps,
        gif_width=gif_width,
        source_size=(template.image_width, template.image_height),
    )
    fixed_rasterizer = None
    rotate_product = name != "orbit_vertical"
    old_fixed_camera = gaussians.fixed_camera
    gaussians.fixed_camera = rotate_product
    try:
        with torch.no_grad():
            for frame_index in tqdm(
                range(frames), desc=name, leave=False
            ):
                fraction = frame_index / frames
                camera = trajectory_camera(
                    name,
                    fraction,
                    template,
                    center,
                    camera_distance,
                    canonical_z,
                    canonical_front,
                )
                if fixed_rasterizer is None or name == "orbit_vertical":
                    rasterizer = set_rasterizer(
                        camera, gaussians, pipeline, background
                    )
                    if name != "orbit_vertical":
                        fixed_rasterizer = rasterizer
                else:
                    rasterizer = fixed_rasterizer
                angle_value = (
                    direction * 2.0 * math.pi * fraction
                    if rotate_product
                    else 0.0
                )
                angle = torch.tensor(
                    [angle_value], dtype=torch.float32, device="cuda"
                )
                image = render(
                    rasterizer,
                    gaussians,
                    axis=axis,
                    center=center_tensor,
                    angle=angle,
                )["render"]
                rgb = (
                    image.clamp(0, 1)
                    .mul(255)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                writers.append(rgb)
    finally:
        gaussians.fixed_camera = old_fixed_camera
        writers.close()
    return {
        "mp4": mp4_path.name,
        "gif": gif_path.name,
        "poster": poster_path.name,
    }


def render_model(
    model_path,
    report_dir,
    *,
    iteration,
    resolution,
    frames,
    fps,
    gif_fps,
    gif_width,
    overwrite,
):
    cfg = load_cfg(model_path)
    if not getattr(cfg, "fixed_camera", False) or getattr(
        cfg, "multi_camera", False
    ):
        raise ValueError(
            f"only fixed, single-camera models are supported: {model_path}"
        )
    cfg.resolution = resolution
    selected_iteration = (
        latest_iteration(model_path) if iteration == -1 else iteration
    )
    checkpoint = (
        model_path
        / "point_cloud"
        / f"iteration_{selected_iteration}"
        / "point_cloud.ply"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing requested checkpoint: {checkpoint}")

    output_dir = report_dir / "media" / model_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    gaussians = GaussianModel(
        cfg.sh_degree,
        fixed_camera=True,
        multi_camera=False,
        number_of_cameras=1,
        axis_mode=getattr(cfg, "axis_mode", "free"),
        axis_tilt_init_deg=getattr(cfg, "axis_tilt_init_deg", 30.0),
        axis_tilt_min_deg=getattr(cfg, "axis_tilt_min_deg", 0.0),
        axis_tilt_max_deg=getattr(cfg, "axis_tilt_max_deg", 90.0),
        axis_side_limit_deg=getattr(cfg, "axis_side_limit_deg", 5.0),
        center_max_offset=getattr(cfg, "center_max_offset", 0.25),
        center_warmup_iterations=getattr(
            cfg, "center_warmup_iterations", 2000
        ),
    )
    scene = Scene(
        cfg,
        gaussians,
        fixed_camera=True,
        load_iteration=selected_iteration,
        shuffle=False,
        random_init=True,
        multi_camera=False,
        eval_mode=True,
    )
    views = scene.getTrainCameras()
    if not views:
        raise ValueError(f"model has no training camera: {model_path}")
    template = views[0]
    pipeline = SimpleNamespace(debug=getattr(cfg, "debug", False))
    background = torch.tensor(
        [1.0, 1.0, 1.0]
        if cfg.white_background
        else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    axis_tensor = gaussians.get_axis(0)
    center_tensor = gaussians.get_center(0)
    (
        axis,
        center,
        camera_distance,
        canonical_z,
        canonical_front,
        source_elevation,
    ) = camera_frame(gaussians, template)

    media = {}
    for name, _title, _description in TRAJECTORIES:
        files = {
            "mp4": output_dir / f"{name}.mp4",
            "gif": output_dir / f"{name}.gif",
            "poster": output_dir / f"{name}_poster.jpg",
        }
        if not overwrite and all(
            path.is_file() and path.stat().st_size
            for path in files.values()
        ):
            media[name] = {
                key: path.name for key, path in files.items()
            }
            print(f"Reusing existing {model_path.name}/{name}")
            continue
        for path in files.values():
            path.unlink(missing_ok=True)
        print(f"Rendering {model_path.name}/{name}")
        media[name] = render_trajectory(
            name,
            output_dir,
            gaussians=gaussians,
            pipeline=pipeline,
            template=template,
            axis=axis_tensor,
            center_tensor=center_tensor,
            center=center,
            camera_distance=camera_distance,
            canonical_z=canonical_z,
            canonical_front=canonical_front,
            background=background,
            direction=int(getattr(cfg, "rotation_direction", 1)),
            frames=frames,
            fps=fps,
            gif_fps=gif_fps,
            gif_width=gif_width,
        )

    product = {
        "name": model_path.name,
        "model_path": str(model_path),
        "iteration": selected_iteration,
        "axis": axis.tolist(),
        "center": center.tolist(),
        "canonical_z": canonical_z.tolist(),
        "source_elevation_deg": source_elevation,
        "camera_distance": camera_distance,
        "width": int(template.image_width),
        "height": int(template.image_height),
        "media": media,
        "rendered_at": datetime.now(timezone.utc).isoformat(),
    }
    del scene, gaussians
    torch.cuda.empty_cache()
    return product


def load_evaluation_summary(model_path, iteration):
    results_path = model_path / "results.json"
    per_view_path = model_path / "per_view.json"
    if not results_path.is_file() or not per_view_path.is_file():
        return {}

    method = f"ours_{iteration}"
    results = json.loads(results_path.read_text()).get(method)
    per_view = json.loads(per_view_path.read_text()).get(method)
    if not results or not per_view:
        return {}

    psnr_values = list(per_view.get("PSNR", {}).values())
    lpips_values = list(per_view.get("LPIPS", {}).values())
    return {
        "psnr": float(results["PSNR"]),
        "ssim": float(results["SSIM"]),
        "lpips": float(results["LPIPS"]),
        "worst_psnr": min(psnr_values) if psnr_values else None,
        "worst_lpips": max(lpips_values) if lpips_values else None,
        "test_views": len(psnr_values),
    }


def saved_gaussian_count(model_path, iteration):
    checkpoint = (
        model_path
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )
    if not checkpoint.is_file():
        return None
    with checkpoint.open("rb") as ply:
        for raw_line in ply:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[-1])
            if line == "end_header":
                break
    raise ValueError(f"PLY vertex count not found: {checkpoint}")


def load_preprocessing_metadata(cfg):
    path = Path(cfg.source_path) / "preprocessing_metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _timestamp_from_log_line(line):
    return datetime.strptime(
        line[:23], "%Y-%m-%d %H:%M:%S,%f"
    ).replace(tzinfo=timezone.utc)


def load_run_timing(model_path, iteration, cfg):
    run_dirs = sorted((model_path / "wandb").glob("run-*"))
    if run_dirs:
        run_dir = run_dirs[-1]
        debug_path = run_dir / "logs" / "debug.log"
        summary_path = run_dir / "files" / "wandb-summary.json"
        output_path = run_dir / "files" / "output.log"
        start = end = None
        if debug_path.is_file():
            for line in debug_path.read_text(errors="replace").splitlines():
                if "run started, returning control" in line:
                    start = _timestamp_from_log_line(line)
                elif "got exitcode: 0" in line:
                    end = _timestamp_from_log_line(line)
        summary = (
            json.loads(summary_path.read_text())
            if summary_path.is_file()
            else {}
        )
        duration = summary.get(
            "_runtime", summary.get("_wandb", {}).get("runtime")
        )
        if start is not None and end is None and duration is not None:
            end = datetime.fromtimestamp(
                start.timestamp() + duration, tz=timezone.utc
            )
        url = None
        if output_path.is_file():
            first_line = output_path.read_text(
                errors="replace"
            ).splitlines()[0]
            if first_line.startswith("W&B run: "):
                url = first_line.removeprefix("W&B run: ").strip()
        if url is None:
            run_id = run_dir.name.rsplit("-", 1)[-1]
            entity = getattr(cfg, "wandb_entity", None)
            project = getattr(cfg, "wandb_project", None)
            if entity and project:
                url = (
                    f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
                )
        return {
            "start": start,
            "end": end,
            "duration_seconds": duration,
            "exact": True,
            "wandb_url": url,
        }

    checkpoints = []
    for checkpoint in (model_path / "point_cloud").glob(
        "iteration_*/point_cloud.ply"
    ):
        try:
            checkpoint_iteration = int(
                checkpoint.parent.name.removeprefix("iteration_")
            )
        except ValueError:
            continue
        checkpoints.append((checkpoint_iteration, checkpoint))
    checkpoints.sort()
    final_path = model_path / "point_cloud" / (
        f"iteration_{iteration}"
    ) / "point_cloud.ply"
    if not checkpoints or not final_path.is_file():
        return {}
    start = datetime.fromtimestamp(
        checkpoints[0][1].stat().st_mtime, tz=timezone.utc
    )
    end = datetime.fromtimestamp(
        final_path.stat().st_mtime, tz=timezone.utc
    )
    return {
        "start": start,
        "end": end,
        "duration_seconds": max(0, (end - start).total_seconds()),
        "exact": False,
        "wandb_url": None,
    }


def format_duration(seconds):
    if seconds is None:
        return "n/a"
    total_seconds = round(float(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {seconds:02d} s"
    return f"{minutes} min {seconds:02d} s"


def format_timestamp(value):
    if value is None:
        return "n/a"
    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def relative_media_path(product, filename):
    return f"media/{product['name']}/{filename}"


def write_report(report_dir, manifest):
    products = sorted(
        manifest["products"].values(),
        key=lambda item: item["name"].lower(),
    )
    product_details = {}
    configurations = []
    preprocessing_records = []
    for product in products:
        model_path = Path(product["model_path"])
        cfg = load_cfg(model_path)
        preprocessing = load_preprocessing_metadata(cfg)
        configurations.append(cfg)
        if preprocessing:
            preprocessing_records.append(preprocessing)
        product_details[product["name"]] = {
            "evaluation": load_evaluation_summary(
                model_path, product["iteration"]
            ),
            "gaussian_count": saved_gaussian_count(
                model_path, product["iteration"]
            ),
            "run_timing": load_run_timing(
                model_path, product["iteration"], cfg
            ),
        }

    def config_values(name, default=None):
        return sorted(
            {getattr(cfg, name, default) for cfg in configurations},
            key=str,
        )

    def joined(values, formatter=str):
        return " / ".join(formatter(value) for value in values)

    image_counts = sorted(
        {item["image_count"] for item in preprocessing_records}
    )
    output_sizes = sorted(
        {item["output_size"][0] for item in preprocessing_records}
    )
    alpha_thresholds = sorted(
        {
            item["segmentation"]["background_threshold_lab"]
            for item in preprocessing_records
        }
    )
    calibrations = sorted(
        {
            Path(item["undistortion"]["calibration_file"]).parent.name
            for item in preprocessing_records
        }
    )
    angle_sets = [
        item["angles_degrees"] for item in preprocessing_records
    ]
    angle_min = min(min(angles) for angles in angle_sets)
    angle_max = max(max(angles) for angles in angle_sets)
    angle_step = angle_sets[0][1] - angle_sets[0][0]
    report_settings = manifest.get("settings", {})

    treatment = f"""
    <section class="overview">
      <header><p class="eyebrow">Method summary</p><h2>Dataset treatment and configuration</h2></header>
      <div class="overview-grid">
        <article class="overview-card">
          <h3>Dataset treatment</h3>
          <ol class="pipeline">
            <li><strong>PNG input.</strong> {joined(image_counts)} RGBA turntable frames per dataset, angle-sorted from {angle_min}° to {angle_max}° in {angle_step}° steps and renamed sequentially.</li>
            <li><strong>Calibration.</strong> Lens undistortion with {html.escape(joined(calibrations))}, OpenCV linear interpolation, and the calibrated intrinsics retained.</li>
            <li><strong>ROI and crop.</strong> A reviewed product ROI determines a centered smallest-square crop; native square outputs span {min(output_sizes):,}–{max(output_sizes):,} px.</li>
            <li><strong>Alpha.</strong> Alpha is regenerated after undistortion using Lab-background segmentation (dataset-specific thresholds {joined(alpha_thresholds, lambda value: f"{value:g}")}; segmentation width 1,600 px), including soft boundary pixels.</li>
            <li><strong>Training inputs.</strong> Full-resolution cropped PNGs, adjusted camera intrinsics, white background, and alpha-aware supervision.</li>
          </ol>
        </article>
        <article class="overview-card">
          <h3>Training configuration</h3>
          <dl class="settings-grid">
            <dt>Optimization</dt><dd>{joined(config_values("iterations"), lambda value: f"{value:,}")} iterations</dd>
            <dt>Representation</dt><dd>3D Gaussians, SH degree {joined(config_values("sh_degree"))}</dd>
            <dt>Camera model</dt><dd>Fixed single camera; full input resolution</dd>
            <dt>Initialization</dt><dd>Random point cloud</dd>
            <dt>Rotation</dt><dd>Counter-clockwise; learned axis and center</dd>
            <dt>Axis constraint</dt><dd>{html.escape(joined(config_values("axis_mode")))}, 0°–90° tilt, ±5° side limit</dd>
            <dt>Motion extras</dt><dd>Residual angle correction off; optical-flow loss off</dd>
            <dt>Densification</dt><dd>iterations {joined(config_values("densify_from_iter"), lambda value: f"{value:,}")}–{joined(config_values("densify_until_iter"), lambda value: f"{value:,}")}, every {joined(config_values("densification_interval"))} steps</dd>
            <dt>Opacity reset</dt><dd>every {joined(config_values("opacity_reset_interval"), lambda value: f"{value:,}")} steps</dd>
            <dt>Gaussian cap</dt><dd>{joined(config_values("max_gaussians", 450000), lambda value: f"{value:,}")}</dd>
            <dt>Loss weights</dt><dd>foreground RGB 1.0; DSSIM 0.2; full RGB 0.1; alpha 0.1; center 0.01</dd>
            <dt>Evaluation</dt><dd>10 held-out views at iteration 30,000</dd>
          </dl>
        </article>
        <article class="overview-card">
          <h3>Report rendering</h3>
          <dl class="settings-grid">
            <dt>Trajectories</dt><dd>{len(TRAJECTORIES)} per product</dd>
            <dt>Video</dt><dd>{report_settings.get("frames", 240)} frames at {report_settings.get("fps", 30)} fps, H.264</dd>
            <dt>GIF</dt><dd>{report_settings.get("gif_fps", 15)} fps, {report_settings.get("gif_width", 480)} px wide</dd>
            <dt>Render scale</dt><dd>1/{report_settings.get("resolution_divisor", 4)} source resolution</dd>
          </dl>
          <p class="metric-note">Quality metrics use the 10 held-out test views. Higher is better for PSNR and SSIM; lower is better for LPIPS.</p>
        </article>
      </div>
    </section>"""

    cards = []
    for product in products:
        videos = []
        for key, title, description in TRAJECTORIES:
            media = product.get("media", {}).get(key)
            if not media:
                continue
            mp4 = html.escape(
                relative_media_path(product, media["mp4"]), quote=True
            )
            gif = html.escape(
                relative_media_path(product, media["gif"]), quote=True
            )
            poster = html.escape(
                relative_media_path(product, media["poster"]), quote=True
            )
            videos.append(
                f"""
                <article class="video-card">
                  <h3>{html.escape(title)}</h3>
                  <video controls loop preload="metadata" playsinline poster="{poster}">
                    <source src="{mp4}" type="video/mp4">
                    Your browser cannot play this video. <a href="{mp4}">Download MP4</a>.
                  </video>
                  <p>{html.escape(description)}</p>
                  <div class="links"><a href="{mp4}" download>MP4</a><a href="{gif}" target="_blank">GIF</a></div>
                </article>"""
            )
        axis = ", ".join(
            f"{value:.4f}" for value in product["axis"]
        )
        center = ", ".join(
            f"{value:.4f}" for value in product["center"]
        )
        details = product_details[product["name"]]
        evaluation = details["evaluation"]

        def metric_card(label, key, digits, unit=""):
            value = evaluation.get(key)
            display = (
                "n/a"
                if value is None
                else f"{value:.{digits}f}{unit}"
            )
            return (
                '<div class="metric"><span>'
                + html.escape(label)
                + "</span><strong>"
                + display
                + "</strong></div>"
            )

        metrics_html = "".join(
            (
                metric_card("Mean PSNR ↑", "psnr", 2, " dB"),
                metric_card("Mean SSIM ↑", "ssim", 4),
                metric_card("Mean LPIPS ↓", "lpips", 4),
                metric_card(
                    "Worst-view PSNR ↑", "worst_psnr", 2, " dB"
                ),
                metric_card(
                    "Worst-view LPIPS ↓", "worst_lpips", 4
                ),
                (
                    '<div class="metric gaussian-count"><span>Saved '
                    'Gaussians</span><strong>'
                    + (
                        f'{details["gaussian_count"]:,}'
                        if details["gaussian_count"] is not None
                        else "n/a"
                    )
                    + "</strong></div>"
                ),
            )
        )
        timing = details["run_timing"]
        timing_label = (
            "Training run"
            if timing.get("exact")
            else "Recorded checkpoint window"
        )
        duration_prefix = "" if timing.get("exact") else "≥ "
        wandb_url = timing.get("wandb_url")
        wandb_link = (
            '<a href="'
            + html.escape(wandb_url, quote=True)
            + '" target="_blank" rel="noopener noreferrer">'
            + "Open W&amp;B run ↗</a>"
            if wandb_url
            else '<span class="unavailable">Not recorded</span>'
        )
        run_html = f"""
          <div class="run-strip">
            <div><span>{timing_label}</span><strong>{format_timestamp(timing.get('start'))} → {format_timestamp(timing.get('end'))}</strong></div>
            <div><span>Duration</span><strong>{duration_prefix}{format_duration(timing.get('duration_seconds'))}</strong></div>
            <div><span>Tracking</span><strong>{wandb_link}</strong></div>
          </div>"""
        cards.append(
            f"""
            <section class="product" data-search="{html.escape(product['name'].lower(), quote=True)}">
              <header>
                <div><h2>{html.escape(product['name'])}</h2><p class="path">{html.escape(product['model_path'])}</p></div>
                <div class="product-actions">
                  <span class="checkpoint">iteration {product['iteration']}</span>
                  <button class="play-toggle product-play-toggle" type="button" aria-pressed="false">Play product videos</button>
                </div>
              </header>
              {run_html}
              <div class="metric-grid">{metrics_html}</div>
              <p class="metric-context">Held-out evaluation across {evaluation.get('test_views', 0)} views at iteration {product['iteration']}.</p>
              <details><summary>Render metadata</summary>
                <dl><dt>Learned axis</dt><dd>[{axis}]</dd><dt>Rotation center</dt><dd>[{center}]</dd>
                <dt>Training elevation</dt><dd>{product['source_elevation_deg']:.2f}°</dd>
                <dt>Resolution</dt><dd>{product['width']} × {product['height']}</dd></dl>
              </details>
              <div class="video-grid">{''.join(videos)}</div>
            </section>"""
        )

    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RotGS Render Report</title><link rel="stylesheet" href="report.css"></head>
<body><main><header class="hero"><div><p class="eyebrow">RotGS · {html.escape(manifest['session_name'])}</p>
<h1>Product render report</h1><p>{len(products)} products · four camera trajectories per product</p></div>
<div class="hero-controls"><label>Filter products<input id="filter" type="search" placeholder="Product name…"></label>
<button id="play-all" class="play-toggle" type="button" aria-pressed="false">Play all videos</button></div></header>
{treatment}
<div id="empty" hidden>No products match the filter.</div>{''.join(cards)}</main>
<script>
const filter = document.querySelector('#filter');
const products = [...document.querySelectorAll('.product')];
const empty = document.querySelector('#empty');
const playAll = document.querySelector('#play-all');
const allVideos = [...document.querySelectorAll('video')];
const isPlaying = video => !video.paused && !video.ended;

function syncButtons() {{
  const anyPlaying = allVideos.some(isPlaying);
  playAll.textContent = anyPlaying ? 'Pause all videos' : 'Play all videos';
  playAll.setAttribute('aria-pressed', String(anyPlaying));
  products.forEach(product => {{
    const button = product.querySelector('.product-play-toggle');
    const productVideos = [...product.querySelectorAll('video')];
    const productPlaying = productVideos.some(isPlaying);
    button.textContent = productPlaying ? 'Pause product videos' : 'Play product videos';
    button.setAttribute('aria-pressed', String(productPlaying));
  }});
}}

function setPlayback(videos, shouldPlay) {{
  videos.forEach(video => {{
    if (shouldPlay) video.play().catch(() => {{}});
    else video.pause();
  }});
  syncButtons();
}}

filter.addEventListener('input', () => {{
  const query = filter.value.trim().toLowerCase();
  let visible = 0;
  products.forEach(product => {{
    const show = product.dataset.search.includes(query);
    product.hidden = !show;
    visible += show;
  }});
  empty.hidden = visible !== 0;
}});

playAll.addEventListener('click', () => {{
  setPlayback(allVideos, !allVideos.some(isPlaying));
}});

products.forEach(product => {{
  const button = product.querySelector('.product-play-toggle');
  const videos = [...product.querySelectorAll('video')];
  button.addEventListener('click', () => {{
    setPlayback(videos, !videos.some(isPlaying));
  }});
}});

allVideos.forEach(video => {{
  video.addEventListener('play', syncButtons);
  video.addEventListener('pause', syncButtons);
  video.addEventListener('ended', syncButtons);
}});
syncButtons();
</script>
</body></html>"""
    css = """
:root{color-scheme:dark;--bg:#0b0d10;--panel:#151920;--soft:#929cab;--line:#29303a;--accent:#75b8ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#f5f7fa;font:15px/1.5 system-ui,sans-serif}
main{width:min(1500px,calc(100% - 32px));margin:auto;padding:40px 0 80px}.hero,.product>header{display:flex;justify-content:space-between;gap:24px;align-items:end}
.hero{margin-bottom:34px}.eyebrow{color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing:.12em}h1{font-size:clamp(32px,5vw,58px);line-height:1;margin:.2em 0}h2{margin:.1em 0;font-size:24px}h3{margin:0 0 12px}
label{color:var(--soft)}input{display:block;margin-top:7px;width:min(360px,80vw);padding:11px 13px;border:1px solid var(--line);border-radius:8px;background:#0f1217;color:white}
.hero-controls,.product-actions{display:flex;gap:12px;align-items:end}.hero-controls{flex-direction:column}.product-actions{align-items:center}.play-toggle{border:1px solid #438fdc;border-radius:8px;background:#153e67;color:#fff;padding:10px 14px;font:inherit;font-weight:700;cursor:pointer;white-space:nowrap}.play-toggle:hover{background:#1d5184}.play-toggle[aria-pressed="true"]{border-color:#d49b45;background:#68471a}
.overview{margin:0 0 38px;padding:24px;background:linear-gradient(145deg,#121a24,#101419);border:1px solid #304158;border-radius:14px}.overview>header h2{margin:.15em 0 22px;font-size:28px}.overview-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.overview-card{padding:18px;background:#0f1217;border:1px solid var(--line);border-radius:10px}.pipeline{margin:0;padding-left:21px}.pipeline li+li{margin-top:9px}.settings-grid{grid-template-columns:minmax(115px,max-content) 1fr;margin:0}.metric-note{margin:18px 0 0;color:var(--soft)}
.product{margin:28px 0;padding:24px;background:var(--panel);border:1px solid var(--line);border-radius:14px}.path{margin:.4em 0;color:var(--soft);overflow-wrap:anywhere}.checkpoint{white-space:nowrap;color:var(--accent)}
.run-strip{display:grid;grid-template-columns:minmax(0,2fr) minmax(150px,.7fr) minmax(140px,.6fr);gap:10px;margin-top:20px}.run-strip>div{padding:11px 12px;background:#101820;border:1px solid var(--line);border-radius:8px}.run-strip span{display:block;color:var(--soft);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.run-strip strong{display:block;margin-top:4px;font-size:14px}.run-strip a{color:var(--accent);text-decoration:none}.run-strip .unavailable{font-size:14px;text-transform:none;letter-spacing:0}
.metric-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-top:20px}.metric{padding:12px;background:#0f1217;border:1px solid var(--line);border-radius:8px}.metric span{display:block;color:var(--soft);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.metric strong{display:block;margin-top:4px;font:700 20px/1.2 ui-monospace,monospace}.gaussian-count{border-color:#365f89}.metric-context{margin:8px 0 0;color:var(--soft);font-size:13px}
details{margin:16px 0}summary{cursor:pointer;color:var(--soft)}dl{display:grid;grid-template-columns:max-content 1fr;gap:5px 14px}dt{color:var(--soft)}dd{margin:0;font-family:ui-monospace,monospace}
.video-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.video-card{min-width:0;padding:16px;background:#0f1217;border-radius:10px}.video-card video{display:block;width:100%;aspect-ratio:1;background:#fff;border-radius:7px}.video-card p{min-height:3em;color:var(--soft)}.links{display:flex;gap:12px}.links a{color:var(--accent);text-decoration:none;font-weight:700}#empty{padding:60px;text-align:center;color:var(--soft)}
@media(max-width:1200px){.overview-grid{grid-template-columns:1fr 1fr}.overview-card:last-child{grid-column:1/-1}.metric-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:800px){.hero,.product>header{align-items:start;flex-direction:column}.hero-controls{align-items:start}.overview{padding:15px}.overview-grid{grid-template-columns:1fr}.overview-card:last-child{grid-column:auto}.run-strip{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.video-grid{grid-template-columns:1fr}.product{padding:15px}main{width:min(100% - 18px,1500px);padding-top:24px}}
""".strip()
    (report_dir / "index.html").write_text(index)
    (report_dir / "report.css").write_text(css + "\n")


def load_manifest(report_dir):
    path = report_dir / "report_manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"unsupported report manifest schema: {path}"
            )
        return manifest
    return {"schema_version": 1, "products": {}}


def infer_report_dir(models):
    parents = {model.parent.resolve() for model in models}
    if len(parents) != 1:
        raise ValueError(
            "models do not share one session directory; "
            "pass --report-dir explicitly"
        )
    return next(iter(parents)) / "render_report"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=Path,
        required=True,
        help="RotGS run path; repeat for multiple runs",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="existing/new report folder; defaults to SESSION/render_report",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="checkpoint iteration; -1 selects latest",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=4,
        choices=(1, 2, 4, 8),
        help="source-resolution divisor",
    )
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-width", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if (
        args.frames < 2
        or args.fps <= 0
        or args.gif_fps <= 0
        or args.gif_width <= 0
    ):
        parser.error(
            "frames must be >= 2 and FPS/width values must be positive"
        )
    models = [path.expanduser().resolve() for path in args.model]
    for model in models:
        if not model.is_dir():
            parser.error(f"model path is not a directory: {model}")
    report_dir = (
        args.report_dir or infer_report_dir(models)
    ).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "media").mkdir(exist_ok=True)
    safe_state(args.quiet)

    manifest = load_manifest(report_dir)
    manifest["session_name"] = report_dir.parent.name
    manifest["report_dir"] = str(report_dir)
    manifest["settings"] = {
        "frames": args.frames,
        "fps": args.fps,
        "gif_fps": args.gif_fps,
        "gif_width": args.gif_width,
        "resolution_divisor": args.resolution,
    }
    for model_path in models:
        product = render_model(
            model_path,
            report_dir,
            iteration=args.iteration,
            resolution=args.resolution,
            frames=args.frames,
            fps=args.fps,
            gif_fps=args.gif_fps,
            gif_width=args.gif_width,
            overwrite=args.overwrite,
        )
        manifest["products"][product["name"]] = product
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        (report_dir / "report_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        write_report(report_dir, manifest)

    print(f"Report written to: {report_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
