#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

import math
import os
from pathlib import Path
import re
import sys
import uuid
from argparse import ArgumentParser, Namespace
from datetime import datetime

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render, set_rasterizer
from scene import GaussianModel, Scene
from scene.residual_predictor import ResidualPredictor
from utils.experiment_tracking import ExperimentTracker
from utils.general_utils import (
    plot_axis,
    plot_point_cloud,
    safe_state,
    save_comparison_image,
)
from utils.image_utils import psnr
from utils.loss_utils import (
    l1_loss,
    masked_l1_loss,
    masked_psnr,
    silhouette_iou_loss,
    ssim,
)
from utils.multi_camera_dataset import (
    group_cameras_by_index,
    load_multi_camera_bundle,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)


def infer_dataset_session(source_path):
    for component in reversed(re.split(r"[\\/]+", os.path.normpath(source_path))):
        match = re.fullmatch(r"session[_ -]?0*(\d+)", component, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def rotation_angle_for_view(viewpoint, residual_predictor, use_local_residual):
    coarse_angle = viewpoint.rotation_angle.unsqueeze(0).to(device)
    time_tensor = torch.as_tensor(
        viewpoint.time, dtype=torch.float32, device=device
    )
    correction = residual_predictor.angle_correction(
        time_tensor,
        viewpoint.cam_idx,
        use_local_residual=use_local_residual,
    )
    return coarse_angle + correction


def foreground_crop(image, target, bbox, padding):
    x0, y0, x1, y1 = bbox
    height, width = image.shape[-2:]
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width, x1 + padding)
    y1 = min(height, y1 + padding)
    return image[..., y0:y1, x0:x1], target[..., y0:y1, x0:x1]


def resolve_geometry_ply(path):
    """Resolve a PLY, an iteration directory, or a completed run directory."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        if candidate.suffix.lower() != ".ply":
            raise ValueError(f"geometry initializer is not a PLY: {candidate}")
        return candidate

    direct = candidate / "point_cloud.ply"
    if direct.is_file():
        return direct

    point_cloud_root = candidate / "point_cloud"
    if point_cloud_root.is_dir():
        iterations = []
        for directory in point_cloud_root.glob("iteration_*"):
            try:
                iteration = int(directory.name.removeprefix("iteration_"))
            except ValueError:
                continue
            ply = directory / "point_cloud.ply"
            if ply.is_file():
                iterations.append((iteration, ply))
        if iterations:
            return max(iterations, key=lambda item: item[0])[1]

    raise FileNotFoundError(
        f"could not find point_cloud.ply below geometry initializer: {candidate}"
    )


def resolve_motion_initializer(path):
    """Resolve a calibration run or iteration directory to its motion state."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        if candidate.name != "motion.json":
            raise ValueError(f"motion initializer is not motion.json: {candidate}")
        motion_path = candidate
    elif (candidate / "motion.json").is_file():
        motion_path = candidate / "motion.json"
    else:
        point_cloud_root = candidate / "point_cloud"
        iterations = []
        if point_cloud_root.is_dir():
            for directory in point_cloud_root.glob("iteration_*"):
                try:
                    iteration = int(directory.name.removeprefix("iteration_"))
                except ValueError:
                    continue
                motion = directory / "motion.json"
                if motion.is_file():
                    iterations.append((iteration, motion))
        if not iterations:
            raise FileNotFoundError(
                f"could not find motion.json below motion initializer: {candidate}"
            )
        _, motion_path = max(iterations, key=lambda item: item[0])

    iteration_directory = motion_path.parent
    try:
        iteration = int(iteration_directory.name.removeprefix("iteration_"))
    except ValueError as error:
        raise ValueError(
            "motion.json must be inside a point_cloud/iteration_N directory"
        ) from error
    run_directory = iteration_directory.parent.parent
    residual_path = (
        run_directory
        / "residual_predictor"
        / f"iteration_{iteration}"
        / "residual_predictor.pth"
    )
    if not residual_path.is_file():
        raise FileNotFoundError(
            "could not find residual predictor matching motion iteration "
            f"{iteration}: {residual_path}"
        )
    return motion_path, residual_path, iteration


def select_reference_camera(bundle, requested_index):
    if requested_index >= 0:
        if requested_index >= len(bundle.passes):
            raise ValueError(
                f"reference camera {requested_index} is outside "
                f"0..{len(bundle.passes) - 1}"
            )
        return requested_index

    elevations = bundle.rough_elevations_degrees
    if elevations is None:
        return len(bundle.passes) // 2
    # The middle/elevated view has enough side and top texture to bootstrap a
    # stable complete cloud, and matches the successful 30-degree single run.
    return min(range(len(elevations)), key=lambda idx: abs(elevations[idx] - 30.0))


def stage_for_iteration(
    iteration,
    bootstrap_iterations,
    pose_warmup_iterations,
    appearance_warmup_iterations=0,
):
    if iteration <= bootstrap_iterations:
        return "bootstrap"
    if iteration <= bootstrap_iterations + pose_warmup_iterations:
        return "pose_alignment"
    if iteration <= (
        bootstrap_iterations
        + pose_warmup_iterations
        + appearance_warmup_iterations
    ):
        return "appearance_alignment"
    return "joint"


def geometry_iteration_for(
    iteration,
    bootstrap_iterations,
    pose_warmup_iterations,
    appearance_warmup_iterations=0,
):
    if iteration <= bootstrap_iterations:
        return iteration
    alignment_end = (
        bootstrap_iterations
        + pose_warmup_iterations
        + appearance_warmup_iterations
    )
    if iteration <= alignment_end:
        return bootstrap_iterations
    return iteration - pose_warmup_iterations - appearance_warmup_iterations


def validate_training_options(args, opt, has_geometry_initializer):
    stage_counts = (
        args.bootstrap_iterations,
        args.pose_warmup_iterations,
        args.appearance_warmup_iterations,
    )
    if any(count < 0 for count in stage_counts):
        raise ValueError("stage iteration counts must be greater than or equal to zero")
    if sum(stage_counts) > opt.iterations:
        raise ValueError(
            "bootstrap, pose, and appearance stage iterations must not exceed "
            "total --iterations"
        )
    if args.init_geometry and args.start_checkpoint:
        raise ValueError("--init_geometry and --start_checkpoint are mutually exclusive")
    if args.init_motion and args.start_checkpoint:
        raise ValueError("--init_motion and --start_checkpoint are mutually exclusive")
    if (
        not has_geometry_initializer
        and not args.start_checkpoint
        and args.bootstrap_iterations == 0
    ):
        warnings.warn(
            "Training from random geometry without a reference-camera bootstrap "
            "is supported but is unlikely to align the cameras. Set "
            "--bootstrap_iterations to about 5000.",
            stacklevel=2,
        )
    if args.max_phase_offset_deg < 0:
        raise ValueError("max_phase_offset_deg must be greater than or equal to zero")

    nonnegative_options = {
        "lambda_foreground_rgb": opt.lambda_foreground_rgb,
        "lambda_full_rgb": opt.lambda_full_rgb,
        "lambda_alpha": opt.lambda_alpha,
        "lambda_silhouette": opt.lambda_silhouette,
        "lambda_center_reg": opt.lambda_center_reg,
        "lambda_axis_side_reg": args.lambda_axis_side_reg,
        "lambda_phase_reg": args.lambda_phase_reg,
        "lambda_sweep_reg": args.lambda_sweep_reg,
        "lambda_depth_reg": args.lambda_depth_reg,
        "depth_max_offset": args.depth_max_offset,
        "depth_warmup_iterations": args.depth_warmup_iterations,
        "topology_warmup_iterations": args.topology_warmup_iterations,
        "prune_min_opacity": args.prune_min_opacity,
        "max_gaussians": args.max_gaussians,
    }
    for option_name, option_value in nonnegative_options.items():
        if option_value < 0:
            raise ValueError(f"{option_name} must be greater than or equal to zero")
    if (
        args.axis_tilt_deviation_limit_deg is not None
        and args.axis_tilt_deviation_limit_deg < 0
    ):
        raise ValueError("axis_tilt_deviation_limit_deg must be nonnegative")
    if opt.ssim_crop_padding < 0:
        raise ValueError("ssim_crop_padding must be greater than or equal to zero")


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    config_args,
    tracker,
):
    if not config_args.multi_camera:
        raise ValueError("train_multi.py requires --multi_camera")

    bundle = load_multi_camera_bundle(dataset.source_path)
    number_of_cameras = len(bundle.passes)
    axis_tilt_init_degrees = bundle.rough_elevations_degrees
    reference_camera_index = select_reference_camera(
        bundle, config_args.reference_camera_index
    )
    config_args.reference_camera_index = reference_camera_index
    geometry_ply = (
        resolve_geometry_ply(config_args.init_geometry)
        if config_args.init_geometry
        else None
    )
    motion_initializer = (
        resolve_motion_initializer(config_args.init_motion)
        if config_args.init_motion
        else None
    )
    validate_training_options(config_args, opt, geometry_ply is not None)

    print(
        "Initializing per-camera rotation axes from rough elevations: "
        + (
            ", ".join(f"{value:g}°" for value in axis_tilt_init_degrees)
            if axis_tilt_init_degrees is not None
            else "metadata unavailable"
        )
    )
    print(
        f"Reference camera: {reference_camera_index}; stages: "
        f"bootstrap={config_args.bootstrap_iterations}, "
        f"pose-only={config_args.pose_warmup_iterations}, "
        f"appearance-only={config_args.appearance_warmup_iterations}, "
        f"joint={opt.iterations - config_args.bootstrap_iterations - config_args.pose_warmup_iterations - config_args.appearance_warmup_iterations}"
    )

    first_iter = 0
    best_foreground_psnr = tracker.get_summary(
        "best/eval_test_foreground_psnr",
        float("-inf"),
    )
    gaussians = GaussianModel(
        dataset.sh_degree,
        optimizer_type=opt.optimizer_type,
        fixed_camera=config_args.fixed_camera,
        wo_axis=config_args.wo_axis,
        multi_camera=True,
        number_of_cameras=number_of_cameras,
        freeze_axis=config_args.freeze_axis or config_args.freeze_motion,
        freeze_center=config_args.freeze_center or config_args.freeze_motion,
        freeze_depth=config_args.freeze_depth or config_args.freeze_motion,
        axis_mode=config_args.axis_mode,
        axis_tilt_init_deg=config_args.axis_tilt_init_deg,
        axis_tilt_min_deg=config_args.axis_tilt_min_deg,
        axis_tilt_max_deg=config_args.axis_tilt_max_deg,
        axis_side_limit_deg=config_args.axis_side_limit_deg,
        axis_tilt_deviation_limit_deg=config_args.axis_tilt_deviation_limit_deg,
        center_max_offset=config_args.center_max_offset,
        center_warmup_iterations=config_args.center_warmup_iterations,
        depth_max_offset=config_args.depth_max_offset,
        depth_warmup_iterations=config_args.depth_warmup_iterations,
        depth_reference_camera_index=reference_camera_index,
        axis_tilt_init_degrees=axis_tilt_init_degrees,
        multi_camera_transform=config_args.multi_camera_transform,
    )
    scene = Scene(
        dataset,
        gaussians,
        config_args.fixed_camera,
        config_args.random,
        True,
    )

    if geometry_ply is not None:
        gaussians.load_geometry_from_ply(
            geometry_ply,
            canonicalize_for_rigid_multi_camera=(
                config_args.multi_camera_transform == "rigid"
            ),
            source_camera_index=config_args.init_geometry_source_camera_index,
        )
    if motion_initializer is not None:
        motion_path, _, _ = motion_initializer
        gaussians.load_motion_from_json(motion_path)

    gaussians.training_setup(opt)
    residual_predictor = ResidualPredictor(
        number_of_cameras,
        max_residual_angle_deg=config_args.max_residual_angle_deg,
        max_sweep_error_deg=config_args.max_sweep_error_deg,
        max_phase_offset_deg=config_args.max_phase_offset_deg,
        reference_camera_index=reference_camera_index,
    )
    residual_predictor.train_setting(opt)
    if motion_initializer is not None:
        _, residual_path, _ = motion_initializer
        residual_predictor.load_state_dict(
            torch.load(
                residual_path,
                map_location=residual_predictor.residuals.device,
                weights_only=True,
            ),
            strict=False,
        )
    if config_args.freeze_motion:
        residual_predictor.requires_grad_(False)

    if checkpoint:
        model_params, first_iter = torch.load(
            checkpoint, map_location="cuda", weights_only=False
        )
        gaussians.restore(model_params, opt)
        checkpoint_model_path = os.path.dirname(checkpoint)
        try:
            residual_predictor.load_weights(
                checkpoint_model_path, iteration=first_iter
            )
        except (FileNotFoundError, OSError):
            print(
                "No matching residual checkpoint found; initialize angle "
                "corrections from zero"
            )

    if checkpoint:
        geometry_description = f"checkpoint {checkpoint}"
    elif geometry_ply is not None:
        geometry_description = str(geometry_ply)
    else:
        geometry_description = "random point cloud"
    print(f"Geometry initialization: {geometry_description}")
    if motion_initializer is None:
        motion_description = "metadata defaults"
    else:
        motion_path, residual_path, motion_iteration = motion_initializer
        motion_description = (
            f"iteration {motion_iteration} from {motion_path.parent.parent.parent} "
            f"({motion_path.name} + {residual_path.name})"
        )
    print(f"Motion initialization: {motion_description}")
    print(
        "Motion optimization: "
        + ("frozen" if config_args.freeze_motion else "enabled")
    )
    print(
        "Camera transform: "
        f"{config_args.multi_camera_transform}; phase bound: "
        f"+/-{config_args.max_phase_offset_deg:g}°; sweep bound: "
        f"+/-{config_args.max_sweep_error_deg:g}°"
    )
    print(
        "Local residual correction: "
        + ("disabled" if config_args.wo_tiny else "enabled")
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = (
        torch.rand((3), device="cuda")
        if opt.random_background
        else background
    )

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    camera_stacks = group_cameras_by_index(
        scene.getTrainCameras(), number_of_cameras
    )

    ema_loss = 0.0
    ema_foreground_loss = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    previous_stage = None

    diagnostic_iterations = {
        1,
        config_args.bootstrap_iterations,
        config_args.bootstrap_iterations + config_args.pose_warmup_iterations,
        config_args.bootstrap_iterations
        + config_args.pose_warmup_iterations
        + config_args.appearance_warmup_iterations,
        *testing_iterations,
        *saving_iterations,
    }
    diagnostic_iterations.discard(0)

    for iteration in range(first_iter, opt.iterations + 1):
        if (
            iteration > first_iter
            and iteration in testing_iterations
            and iteration in checkpoint_iterations
        ):
            # Validation can require substantially more GPU memory than
            # training. Keep one rolling snapshot of the last completed step.
            recovery_iteration = iteration - 1
            recovery_checkpoint = os.path.join(
                scene.model_path, "chkpnt_pre_validation.pth"
            )
            print(
                f"\n[ITER {recovery_iteration}] Saving pre-validation "
                "recovery checkpoint"
            )
            torch.save(
                (gaussians.capture(), recovery_iteration),
                recovery_checkpoint,
            )
            residual_predictor.save_weights(
                dataset.model_path,
                recovery_iteration,
            )

        iter_start.record()
        stage = stage_for_iteration(
            iteration,
            config_args.bootstrap_iterations,
            config_args.pose_warmup_iterations,
            config_args.appearance_warmup_iterations,
        )
        canonical_frozen = stage == "pose_alignment"
        structure_frozen = stage in {"pose_alignment", "appearance_alignment"}
        geometry_iteration = geometry_iteration_for(
            iteration,
            config_args.bootstrap_iterations,
            config_args.pose_warmup_iterations,
            config_args.appearance_warmup_iterations,
        )
        joint_iteration = max(
            0,
            iteration
            - config_args.bootstrap_iterations
            - config_args.pose_warmup_iterations
            - config_args.appearance_warmup_iterations,
        )
        topology_allowed = (
            not structure_frozen
            and (
                geometry_ply is None
                or stage == "bootstrap"
                or joint_iteration > config_args.topology_warmup_iterations
            )
        )
        gaussians.update_learning_rate(
            iteration,
            geometry_iteration=max(1, geometry_iteration),
            pose_iteration=iteration,
        )
        residual_predictor.update_learning_rate(iteration)

        if stage != previous_stage:
            print(f"\n[ITER {iteration}] Entering {stage} stage")
            previous_stage = stage

        if not structure_frozen and geometry_iteration > 0 and geometry_iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if stage == "bootstrap":
            cam_idx = reference_camera_index
        else:
            multi_iteration = iteration - config_args.bootstrap_iterations - 1
            cam_idx = (
                multi_iteration // max(1, opt.camera_interval)
            ) % number_of_cameras

        if not camera_stacks[cam_idx]:
            fresh_stacks = group_cameras_by_index(
                scene.getTrainCameras(), number_of_cameras
            )
            camera_stacks[cam_idx] = fresh_stacks[cam_idx]
        viewpoint_cam = camera_stacks[cam_idx].pop(0)

        viewpoint_rasterizer = set_rasterizer(
            viewpoint_cam, gaussians, pipe, bg, scaling_modifier=1.0
        )
        gt_image = viewpoint_cam.original_image.cuda()
        angle = rotation_angle_for_view(
            viewpoint_cam,
            residual_predictor,
            use_local_residual=not config_args.wo_tiny,
        )
        axis = gaussians.get_axis(cam_idx)
        center = gaussians.get_center(cam_idx)
        render_pkg = render(
            viewpoint_rasterizer,
            gaussians,
            axis=axis,
            center=center,
            angle=angle,
        )
        image = render_pkg["render"]
        rendered_alpha = render_pkg["rendered_alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        del render_pkg

        foreground_mask = viewpoint_cam.alpha_mask
        foreground_rgb_loss = masked_l1_loss(
            image, gt_image, foreground_mask
        )
        full_rgb_loss = l1_loss(image, gt_image)
        image_crop, gt_crop = foreground_crop(
            image,
            gt_image,
            viewpoint_cam.foreground_bbox,
            opt.ssim_crop_padding,
        )
        foreground_ssim_value = ssim(image_crop, gt_crop)
        alpha_loss = l1_loss(rendered_alpha.squeeze(), foreground_mask)
        silhouette_loss = silhouette_iou_loss(
            rendered_alpha, foreground_mask
        )
        center_reg_loss = gaussians.motion_regularization()
        axis_side_reg_loss = gaussians.axis_side_regularization()
        depth_reg_loss = gaussians.depth_regularization()
        phase_reg_loss, sweep_reg_loss = (
            residual_predictor.correction_regularization()
        )

        image_loss = (
            (1.0 - opt.lambda_dssim)
            * opt.lambda_foreground_rgb
            * foreground_rgb_loss
            + opt.lambda_dssim * (1.0 - foreground_ssim_value)
            + opt.lambda_full_rgb * full_rgb_loss
            + opt.lambda_alpha * alpha_loss
            + opt.lambda_silhouette * silhouette_loss
        )
        loss = (
            image_loss
            + opt.lambda_center_reg * center_reg_loss
            + config_args.lambda_axis_side_reg * axis_side_reg_loss
            + config_args.lambda_phase_reg * phase_reg_loss
            + config_args.lambda_sweep_reg * sweep_reg_loss
            + config_args.lambda_depth_reg * depth_reg_loss
        )
        loss.backward()
        if canonical_frozen:
            gaussians.clear_geometry_gradients()
        elif structure_frozen:
            gaussians.clear_structure_gradients()
        iter_end.record()

        with torch.no_grad():
            is_validation_iteration = iteration in testing_iterations
            is_densification_iteration = (
                topology_allowed
                and geometry_iteration < opt.densify_until_iter
                and geometry_iteration > opt.densify_from_iter
                and geometry_iteration % opt.densification_interval == 0
            )
            is_opacity_reset_iteration = (
                topology_allowed
                and geometry_iteration < opt.densify_until_iter
                and not config_args.disable_opacity_reset
                and (
                    geometry_iteration % opt.opacity_reset_interval == 0
                    or (
                        dataset.white_background
                        and geometry_iteration == opt.densify_from_iter
                    )
                )
            )
            should_log = tracker.enabled and (
                iteration % config_args.wandb_log_interval == 0
                or is_validation_iteration
                or is_densification_iteration
                or is_opacity_reset_iteration
                or iteration == opt.iterations
            )

            if iteration in diagnostic_iterations:
                axes = torch.stack(
                    [gaussians.get_axis(index) for index in range(number_of_cameras)]
                )
                centers = torch.stack(
                    [gaussians.get_center(index) for index in range(number_of_cameras)]
                )
                plot_point_cloud(
                    gaussians.get_xyz,
                    iteration,
                    filename=os.path.join(
                        dataset.model_path,
                        "pointcloud",
                        f"pcd_{iteration}.png",
                    ),
                    axis=axes,
                    centers=centers,
                    number_of_cameras=number_of_cameras,
                )
                save_comparison_image(
                    iteration,
                    image,
                    gt_image,
                    save_path=os.path.join(dataset.model_path, "comparison.png"),
                )
                plot_axis(
                    axes,
                    filename=os.path.join(dataset.model_path, "graph", "axis.png"),
                )
                residual_predictor.plot_residual(dataset.model_path)

            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            ema_foreground_loss = (
                0.4 * foreground_rgb_loss.item()
                + 0.6 * ema_foreground_loss
            )
            if iteration % 10 == 0:
                tilt, side, center_shift = gaussians.motion_summary(cam_idx)
                camera_depth = gaussians.get_camera_depth(cam_idx)
                phase_deg = torch.rad2deg(
                    residual_predictor.effective_phase_offset(cam_idx)
                )
                sweep_deg = torch.rad2deg(
                    residual_predictor.effective_sweep_error(cam_idx)
                )
                progress_bar.set_postfix(
                    {
                        "stage": stage[:5],
                        "cam": cam_idx,
                        "loss": f"{ema_loss:.4f}",
                        "fg": f"{ema_foreground_loss:.4f}",
                        "alpha": f"{alpha_loss.item():.4f}",
                        "sil": f"{silhouette_loss.item():.4f}",
                        "tilt": f"{tilt.item():.2f}",
                        "side": f"{side.item():.2f}",
                        "center": f"{center_shift.item():.4f}",
                        "depth": f"{camera_depth.item():.3f}",
                        "phase": f"{phase_deg.item():.2f}",
                        "sweep": f"{sweep_deg.item():.2f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            metrics = {}
            gaussian_count_before_update = gaussians.get_xyz.shape[0]
            if should_log:
                visible_gaussians = visibility_filter.sum().item()
                metrics.update(
                    {
                        "train/stage": stage,
                        "train/current_camera": cam_idx,
                        "train/loss_total": loss.item(),
                        "train/loss_image": image_loss.item(),
                        "train/raw/foreground_l1": foreground_rgb_loss.item(),
                        "train/raw/full_l1": full_rgb_loss.item(),
                        "train/raw/foreground_dssim": (
                            1.0 - foreground_ssim_value.item()
                        ),
                        "train/raw/alpha_l1": alpha_loss.item(),
                        "train/raw/silhouette_iou_loss": silhouette_loss.item(),
                        "train/raw/axis_side_regularization": axis_side_reg_loss.item(),
                        "train/raw/center_regularization": center_reg_loss.item(),
                        "train/raw/depth_regularization": depth_reg_loss.item(),
                        "train/raw/phase_regularization": phase_reg_loss.item(),
                        "train/raw/sweep_regularization": sweep_reg_loss.item(),
                        "train/weighted/foreground_l1": (
                            (1.0 - opt.lambda_dssim)
                            * opt.lambda_foreground_rgb
                            * foreground_rgb_loss.item()
                        ),
                        "train/weighted/foreground_dssim": (
                            opt.lambda_dssim
                            * (1.0 - foreground_ssim_value.item())
                        ),
                        "train/weighted/full_l1": (
                            opt.lambda_full_rgb * full_rgb_loss.item()
                        ),
                        "train/weighted/alpha_l1": (
                            opt.lambda_alpha * alpha_loss.item()
                        ),
                        "train/weighted/silhouette_iou_loss": (
                            opt.lambda_silhouette * silhouette_loss.item()
                        ),
                        "train/weighted/axis_side_regularization": (
                            config_args.lambda_axis_side_reg
                            * axis_side_reg_loss.item()
                        ),
                        "train/weighted/center_regularization": (
                            opt.lambda_center_reg * center_reg_loss.item()
                        ),
                        "train/weighted/depth_regularization": (
                            config_args.lambda_depth_reg * depth_reg_loss.item()
                        ),
                        "train/weighted/phase_regularization": (
                            config_args.lambda_phase_reg * phase_reg_loss.item()
                        ),
                        "train/weighted/sweep_regularization": (
                            config_args.lambda_sweep_reg * sweep_reg_loss.item()
                        ),
                        "performance/iteration_ms": iter_start.elapsed_time(iter_end),
                        "scene/gaussians": gaussian_count_before_update,
                        "scene/gaussian_limit": config_args.max_gaussians,
                        "scene/visible_gaussians": visible_gaussians,
                        "scene/visible_fraction": (
                            visible_gaussians / max(1, gaussian_count_before_update)
                        ),
                    }
                )
                for group in gaussians.optimizer.param_groups:
                    metrics[f"learning_rate/{group['name']}"] = group["lr"]
                metrics["learning_rate/exposure"] = (
                    gaussians.exposure_optimizer.param_groups[0]["lr"]
                )
                metrics["learning_rate/residual"] = (
                    residual_predictor.optimizer.param_groups[0]["lr"]
                )
                for camera_index in range(number_of_cameras):
                    prefix = f"motion/camera_{camera_index:02d}"
                    camera_tilt, camera_side, camera_center_shift = (
                        gaussians.motion_summary(camera_index)
                    )
                    camera_center = gaussians.get_center(camera_index)
                    local_residuals_deg = torch.rad2deg(
                        residual_predictor._apply_bound(
                            residual_predictor.residuals[camera_index]
                        )
                    )
                    metrics.update(
                        {
                            f"{prefix}/axis_tilt_deg": camera_tilt.item(),
                            f"{prefix}/axis_side_deg": camera_side.item(),
                            f"{prefix}/center_shift": camera_center_shift.item(),
                            f"{prefix}/center_x": camera_center[0].item(),
                            f"{prefix}/center_y": camera_center[1].item(),
                            f"{prefix}/center_z": camera_center[2].item(),
                            f"{prefix}/depth": gaussians.get_camera_depth(
                                camera_index
                            ).item(),
                            f"{prefix}/phase_deg": torch.rad2deg(
                                residual_predictor.effective_phase_offset(
                                    camera_index
                                )
                            ).item(),
                            f"{prefix}/sweep_deg": torch.rad2deg(
                                residual_predictor.effective_sweep_error(
                                    camera_index
                                )
                            ).item(),
                            f"{prefix}/local_residual_mean_abs_deg": (
                                local_residuals_deg.abs().mean().item()
                            ),
                            f"{prefix}/local_residual_max_abs_deg": (
                                local_residuals_deg.abs().max().item()
                            ),
                        }
                    )

            if is_validation_iteration:
                # The topology update below only needs view-space gradients,
                # visibility and radii. Release image/loss tensors before
                # allocating validation renders.
                del (
                    image,
                    rendered_alpha,
                    gt_image,
                    foreground_mask,
                    image_crop,
                    gt_crop,
                    loss,
                    image_loss,
                    foreground_rgb_loss,
                    full_rgb_loss,
                    foreground_ssim_value,
                    alpha_loss,
                    silhouette_loss,
                    center_reg_loss,
                    axis_side_reg_loss,
                    depth_reg_loss,
                    phase_reg_loss,
                    sweep_reg_loss,
                )
                torch.cuda.empty_cache()

            report_metrics, report_media = training_report(
                iteration,
                testing_iterations,
                gaussians,
                scene,
                pipe,
                bg,
                residual_predictor,
                config_args,
                tracker,
            )
            metrics.update(report_metrics)
            metrics.update(report_media)

            test_foreground_psnr = report_metrics.get(
                "eval/test/foreground_psnr"
            )
            is_best = (
                test_foreground_psnr is not None
                and test_foreground_psnr > best_foreground_psnr
            )
            if is_best:
                best_foreground_psnr = test_foreground_psnr
                tracker.set_summary(
                    "best/eval_test_foreground_psnr",
                    best_foreground_psnr,
                )
                tracker.set_summary("best/iteration", iteration)

            needs_best_artifact = (
                tracker.artifact_policy == "best_and_final" and is_best
            )
            should_save = iteration in saving_iterations or needs_best_artifact
            if should_save:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                residual_predictor.save_weights(dataset.model_path, iteration)

            densification_metrics = {}
            if (
                topology_allowed
                and geometry_iteration < opt.densify_until_iter
            ):
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if is_densification_iteration:
                    size_threshold = (
                        20
                        if (
                            not config_args.disable_size_pruning
                            and geometry_iteration > opt.opacity_reset_interval
                        )
                        else None
                    )
                    gaussians_before = gaussians.get_xyz.shape[0]
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        config_args.prune_min_opacity,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        max_gaussians=config_args.max_gaussians,
                    )
                    gaussians_after = gaussians.get_xyz.shape[0]
                    limit_description = (
                        f"/{config_args.max_gaussians}"
                        if config_args.max_gaussians > 0
                        else ""
                    )
                    print(
                        f"\n[ITER {iteration}] Densification: "
                        f"{gaussians_before} -> {gaussians_after}{limit_description}"
                    )
                    densification_metrics.update(
                        {
                            "densification/event": 1,
                            "densification/count_before": gaussians_before,
                            "densification/count_after": gaussians_after,
                            "densification/net_change": (
                                gaussians_after - gaussians_before
                            ),
                            "densification/limit_reached": int(
                                config_args.max_gaussians > 0
                                and gaussians_after >= config_args.max_gaussians
                            ),
                        }
                    )
                if is_opacity_reset_iteration:
                    gaussians.reset_opacity()
                    densification_metrics["densification/opacity_reset"] = 1

            if should_log:
                metrics["scene/gaussians_after_update"] = (
                    gaussians.get_xyz.shape[0]
                )
                if (
                    is_validation_iteration
                    or is_densification_iteration
                    or is_opacity_reset_iteration
                ):
                    opacity = gaussians.get_opacity
                    metrics.update(
                        {
                            "scene/opacity_mean": opacity.mean().item(),
                            "scene/opacity_min": opacity.min().item(),
                            "scene/opacity_max": opacity.max().item(),
                        }
                    )
                metrics.update(densification_metrics)

            if iteration < opt.iterations:
                if not canonical_frozen:
                    gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                residual_predictor.optimizer.step()
                residual_predictor.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + f"/chkpnt{iteration}.pth",
                )
                residual_predictor.save_weights(dataset.model_path, iteration)

            if metrics:
                tracker.log(metrics, iteration)

            artifact_aliases = []
            if tracker.artifact_policy == "best_and_final" and is_best:
                artifact_aliases.append("best")
            if (
                tracker.artifact_policy in ("final", "best_and_final")
                and iteration == opt.iterations
            ):
                artifact_aliases.append("final")
            if artifact_aliases:
                tracker.log_model_artifact(
                    dataset.model_path,
                    iteration,
                    artifact_aliases,
                )


def prepare_output_and_logger(args, output_name="random", config_args=None):
    if output_name == "random":
        if not args.model_path:
            unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[:10])
    else:
        args.model_path = os.path.join("./output/", output_name)

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "pointcloud"), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "graph"), exist_ok=True)
    if config_args is not None:
        config_args.model_path = args.model_path
        config_to_write = config_args
    else:
        config_to_write = args
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_file:
        cfg_log_file.write(str(Namespace(**vars(config_to_write))))


def training_report(
    iteration,
    testing_iterations,
    gaussians,
    scene,
    pipe,
    background,
    residual_predictor,
    args,
    tracker,
):
    report_metrics = {}
    report_media = {}
    if iteration not in testing_iterations:
        return report_metrics, report_media

    torch.cuda.empty_cache()
    train_cameras = scene.getTrainCameras()
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": [
                train_cameras[index % len(train_cameras)]
                for index in range(5, 30, 5)
            ],
        },
    )

    for config in validation_configs:
        cameras = config["cameras"]
        if not cameras:
            continue
        totals = {}
        counts = {}
        comparison_images = []
        for view_index, viewpoint in enumerate(cameras):
            angle = rotation_angle_for_view(
                viewpoint,
                residual_predictor,
                use_local_residual=not args.wo_tiny,
            )
            camera_index = viewpoint.cam_idx
            axis = gaussians.get_axis(camera_index)
            center = gaussians.get_center(camera_index)
            rasterizer = set_rasterizer(
                viewpoint,
                scene.gaussians,
                pipe,
                background,
                scaling_modifier=1.0,
            )
            render_pkg = render(
                rasterizer,
                scene.gaussians,
                axis=axis,
                center=center,
                angle=angle,
            )
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(
                viewpoint.original_image.to("cuda"), 0.0, 1.0
            )
            if args.train_test_exp:
                image = image[..., image.shape[-1] // 2 :]
                gt_image = gt_image[..., gt_image.shape[-1] // 2 :]

            image_crop, gt_crop = foreground_crop(
                image,
                gt_image,
                viewpoint.foreground_bbox,
                args.ssim_crop_padding,
            )
            silhouette_iou = 1.0 - silhouette_iou_loss(
                render_pkg["rendered_alpha"], viewpoint.alpha_mask
            )
            view_metrics = torch.stack(
                (
                    l1_loss(image, gt_image).mean(),
                    psnr(image, gt_image).mean(),
                    masked_psnr(image, gt_image, viewpoint.alpha_mask),
                    ssim(image_crop, gt_crop),
                    silhouette_iou,
                )
            ).double()
            totals[camera_index] = totals.get(
                camera_index, torch.zeros_like(view_metrics)
            ) + view_metrics
            counts[camera_index] = counts.get(camera_index, 0) + 1

            if tracker.enabled and view_index < 3:
                comparison = torch.cat(
                    (gt_image, image, torch.abs(image - gt_image)), dim=-1
                )
                view_name = getattr(viewpoint, "image_name", str(view_index))
                comparison_images.append(
                    tracker.image(
                        comparison,
                        caption=(
                            f"{config['name']} camera {camera_index} view "
                            f"{view_name}: ground truth | render | absolute error"
                        ),
                    )
                )
                del comparison

            del render_pkg, image, gt_image, image_crop, gt_crop

        all_metrics = torch.zeros(5, dtype=torch.float64, device="cuda")
        all_count = 0
        camera_foreground_psnr = []
        for camera_index in sorted(totals):
            camera_metrics = totals[camera_index] / counts[camera_index]
            all_metrics += totals[camera_index]
            all_count += counts[camera_index]
            camera_foreground_psnr.append(camera_metrics[2].item())
            print(
                f"\n[ITER {iteration}] Evaluating {config['name']} "
                f"cam {camera_index}: L1 {camera_metrics[0]:.6f} "
                f"PSNR {camera_metrics[1]:.4f} "
                f"FG_PSNR {camera_metrics[2]:.4f} "
                f"FG_SSIM {camera_metrics[3]:.6f} "
                f"SIL_IOU {camera_metrics[4]:.6f}"
            )
            prefix = f"eval/{config['name']}/camera_{camera_index:02d}"
            report_metrics.update(
                {
                    f"{prefix}/l1": camera_metrics[0].item(),
                    f"{prefix}/psnr": camera_metrics[1].item(),
                    f"{prefix}/foreground_psnr": camera_metrics[2].item(),
                    f"{prefix}/foreground_ssim": camera_metrics[3].item(),
                    f"{prefix}/silhouette_iou": camera_metrics[4].item(),
                }
            )

        all_metrics /= all_count
        print(
            f"[ITER {iteration}] Evaluating {config['name']} all: "
            f"L1 {all_metrics[0]:.6f} PSNR {all_metrics[1]:.4f} "
            f"FG_PSNR {all_metrics[2]:.4f} "
            f"FG_SSIM {all_metrics[3]:.6f} "
            f"SIL_IOU {all_metrics[4]:.6f}"
        )
        prefix = f"eval/{config['name']}"
        report_metrics.update(
            {
                f"{prefix}/l1": all_metrics[0].item(),
                f"{prefix}/psnr": all_metrics[1].item(),
                f"{prefix}/foreground_psnr": all_metrics[2].item(),
                f"{prefix}/foreground_ssim": all_metrics[3].item(),
                f"{prefix}/silhouette_iou": all_metrics[4].item(),
                f"{prefix}/worst_camera_foreground_psnr": min(
                    camera_foreground_psnr
                ),
                f"{prefix}/num_views": all_count,
            }
        )
        if comparison_images:
            report_media[f"media/{config['name']}/comparisons"] = comparison_images

    torch.cuda.empty_cache()
    return report_metrics, report_media

if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-camera training parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[3_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[3_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument(
        "--init_geometry",
        type=str,
        default=None,
        help=(
            "Optional single-camera PLY, iteration directory, or run directory. "
            "It accelerates initialization but is not required."
        ),
    )
    parser.add_argument(
        "--init_motion",
        type=str,
        default=None,
        help=(
            "Optional calibrated multi-camera run, iteration directory, or "
            "motion.json. Imports motion and residual state without geometry."
        ),
    )
    parser.add_argument(
        "--init_geometry_source_camera_index", type=int, default=0
    )
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=5_000,
        help="Reference-camera geometry bootstrap iterations.",
    )
    parser.add_argument(
        "--pose_warmup_iterations",
        type=int,
        default=2_000,
        help="All-camera alignment iterations with canonical geometry frozen.",
    )
    parser.add_argument(
        "--appearance_warmup_iterations",
        type=int,
        default=0,
        help=(
            "All-camera appearance alignment iterations with positions, opacity, "
            "scale, and Gaussian rotations frozen."
        ),
    )
    parser.add_argument(
        "--reference_camera_index",
        type=int,
        default=-1,
        help="Reference pass index; -1 chooses the view nearest 30 degrees.",
    )
    parser.add_argument(
        "--multi_camera_transform",
        choices=("rigid", "legacy"),
        default="rigid",
    )
    parser.add_argument("--fixed_camera", action="store_true", default=True)
    parser.add_argument("--random", action="store_true", default=True)
    parser.add_argument("--sfm", action="store_true", default=False)
    parser.add_argument("--wo_tiny", action="store_true", default=False)
    parser.add_argument("--max_residual_angle_deg", type=float, default=1.0)
    parser.add_argument("--max_sweep_error_deg", type=float, default=4.0)
    parser.add_argument("--max_phase_offset_deg", type=float, default=15.0)
    parser.add_argument("--lambda_axis_side_reg", type=float, default=0.001)
    parser.add_argument("--lambda_phase_reg", type=float, default=0.001)
    parser.add_argument("--lambda_sweep_reg", type=float, default=0.001)
    parser.add_argument("--lambda_depth_reg", type=float, default=0.001)
    parser.add_argument("--wo_flow", action="store_true", default=True)
    parser.add_argument("--wo_UDFS", action="store_true", default=False)
    parser.add_argument("--wo_axis", action="store_true", default=False)
    parser.add_argument("--freeze_axis", action="store_true", default=False)
    parser.add_argument("--freeze_center", action="store_true", default=False)
    parser.add_argument("--freeze_depth", action="store_true", default=False)
    parser.add_argument(
        "--freeze_motion",
        action="store_true",
        default=False,
        help="Freeze imported axes, centers, depths, phase, sweep, and residuals.",
    )
    parser.add_argument(
        "--axis_mode",
        choices=("free", "bounded_tilt"),
        default="bounded_tilt",
    )
    parser.add_argument("--axis_tilt_init_deg", type=float, default=30.0)
    parser.add_argument("--axis_tilt_min_deg", type=float, default=0.0)
    parser.add_argument("--axis_tilt_max_deg", type=float, default=90.0)
    parser.add_argument("--axis_side_limit_deg", type=float, default=5.0)
    parser.add_argument(
        "--axis_tilt_deviation_limit_deg",
        type=float,
        default=None,
        help="Bound each tilt to +/- this many degrees around its metadata elevation.",
    )
    parser.add_argument("--center_max_offset", type=float, default=0.5)
    parser.add_argument("--center_warmup_iterations", type=int, default=2_000)
    parser.add_argument("--depth_max_offset", type=float, default=2.0)
    parser.add_argument("--depth_warmup_iterations", type=int, default=0)
    parser.add_argument(
        "--topology_warmup_iterations",
        type=int,
        default=0,
        help="Delay densification/pruning after an imported cloud enters joint training.",
    )
    parser.add_argument("--prune_min_opacity", type=float, default=0.005)
    parser.add_argument(
        "--max_gaussians",
        type=int,
        default=0,
        help="Maximum Gaussians during densification; 0 leaves it unlimited.",
    )
    parser.add_argument("--disable_opacity_reset", action="store_true")
    parser.add_argument("--disable_size_pruning", action="store_true")
    parser.add_argument(
        "--multi_camera", action="store_true", default=True
    )
    parser.add_argument("--name", type=str, default="exper")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help="portable dataset identifier stored in W&B",
    )
    parser.add_argument(
        "--dataset_session",
        type=int,
        default=None,
        help="numeric capture-session ID stored in W&B; inferred when possible",
    )
    parser.add_argument(
        "--preprocessing_variant",
        type=str,
        default=None,
        help="optional preprocessing label stored in W&B",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="W&B tracking mode; disabled preserves the original path",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "RotGS"),
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.getenv("WANDB_ENTITY", "3D-Scanning-MT"),
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_notes", type=str, default=None)
    parser.add_argument(
        "--wandb_log_interval",
        type=int,
        default=10,
        help="number of training iterations between W&B scalar logs",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help=(
            "existing W&B run ID; inferred from the output folder only for "
            "full checkpoint resumes"
        ),
    )
    parser.add_argument(
        "--wandb_artifacts",
        choices=("none", "final", "best_and_final"),
        default="none",
        help="model artifact upload policy",
    )

    args = parser.parse_args(sys.argv[1:])
    if args.wandb_log_interval < 1:
        parser.error("--wandb_log_interval must be at least 1")
    args.save_iterations.append(args.iterations)

    if args.sfm:
        args.random = False

    dataset = lp.extract(args)
    bundle = load_multi_camera_bundle(dataset.source_path)
    args.reference_camera_index = select_reference_camera(
        bundle,
        args.reference_camera_index,
    )
    args.camera_count = len(bundle.passes)
    args.camera_passes = [item.directory_name for item in bundle.passes]
    args.camera_elevations_degrees = (
        list(bundle.rough_elevations_degrees)
        if bundle.rough_elevations_degrees is not None
        else None
    )
    if args.dataset_id is None:
        args.dataset_id = os.path.basename(os.path.normpath(dataset.source_path))
    if args.dataset_session is None:
        args.dataset_session = infer_dataset_session(dataset.source_path)

    prepare_output_and_logger(
        dataset,
        output_name=args.name,
        config_args=args,
    )
    print("Optimizing " + dataset.model_path)

    # Tracker initialization happens before resetting training RNGs so the
    # external service cannot perturb RotGS's seeded numerical path.
    tracker = ExperimentTracker.create(args, dataset.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    start_time = datetime.now()
    try:
        training(
            dataset,
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args,
            tracker,
        )
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    else:
        tracker.finish()

    end_time = datetime.now()
    print(f"Duration: {end_time - start_time}")
    print("\nTraining complete.")

    with open("duration_log.txt", "a") as duration_log:
        duration_log.write(f"{args.name}: {end_time - start_time}\n")
