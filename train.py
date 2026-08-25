#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
from scene.residual_predictor import ResidualPredictor
import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, masked_l1_loss, masked_psnr, ssim
from gaussian_renderer import render, set_rasterizer
import sys
from scene import Scene, GaussianModel 
from utils.general_utils import safe_state, plot_residual, plot_axis, save_comparison_image, plot_point_cloud, plot_flow
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.loss_utils import compute_flow, L1_flow, cosine_flow, flow_schedule
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from submodules.gmflow.config import get_cfg as get_gmflow_cfg
from submodules.gmflow.gmflow import build_gmflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)


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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):

    number_of_cameras = 1 # single monocular camera system
    cam_idx = 0
    first_iter = 0
    tb_writer = prepare_output_and_logger(
        dataset, output_name=args.name, config_args=args
    )
    gaussians = GaussianModel(
        dataset.sh_degree,
        optimizer_type=opt.optimizer_type,
        fixed_camera=args.fixed_camera,
        wo_axis=args.wo_axis,
        multi_camera=args.multi_camera,
        number_of_cameras=number_of_cameras,
        freeze_axis=args.freeze_axis,
        freeze_center=args.freeze_center,
        axis_mode=args.axis_mode,
        axis_tilt_init_deg=args.axis_tilt_init_deg,
        axis_tilt_min_deg=args.axis_tilt_min_deg,
        axis_tilt_max_deg=args.axis_tilt_max_deg,
        axis_side_limit_deg=args.axis_side_limit_deg,
        center_max_offset=args.center_max_offset,
        center_warmup_iterations=args.center_warmup_iterations,
    )
    scene = Scene(dataset, gaussians, args.fixed_camera, args.random, args.multi_camera)
    gaussians.training_setup(opt)
    xyz_group = next(
        group for group in gaussians.optimizer.param_groups if group["name"] == "xyz"
    )
    print(
        f"Spatial LR scale: {gaussians.spatial_lr_scale:.6g}; "
        f"initial XYZ learning rate: {xyz_group['lr']:.6g}"
    )
    residual_predictor = ResidualPredictor(
        number_of_cameras,
        max_residual_angle_deg=args.max_residual_angle_deg,
        max_sweep_error_deg=args.max_sweep_error_deg,
    )
    residual_predictor.train_setting(opt)

    if not args.wo_flow:
        print("Use optical flow supervision")
        cfg = get_gmflow_cfg()  
        flownet = torch.nn.DataParallel(build_gmflow(cfg)) 
        flownet = flownet.module
        gm_checkpoint = torch.load(cfg.model, map_location = 'cuda')
        weights = gm_checkpoint['model'] if 'model' in gm_checkpoint else gm_checkpoint
        flownet.load_state_dict(weights)
        flownet = flownet.cuda()
        flownet.eval()
        flow_2d_gt_list = []
        flow_alpha_mask_list = []
    else:
        print("without optical flow supervision")

    if not args.wo_tiny:
        print("Use tiny angle to predict residual")
        if args.max_residual_angle_deg > 0:
            print(
                "Bound residual angle correction to "
                f"+/-{args.max_residual_angle_deg:g} degrees"
            )
        else:
            print("Residual angle correction is unbounded")
    else:
        print("without tiny angle to predict residual")

    if args.max_sweep_error_deg > 0:
        print(
            "Learn a linearly accumulated sweep correction bounded to "
            f"+/-{args.max_sweep_error_deg:g} degrees"
        )
    else:
        print("without learned sweep correction")

    print(
        "Rotation axis: "
        + ("fixed" if gaussians.freeze_axis else f"learned ({gaussians.axis_mode})")
        + "; rotation center: "
        + ("fixed" if gaussians.freeze_center else "learned")
    )

    if args.flow_downsampling < 1:
        raise ValueError("flow_downsampling must be an integer greater than or equal to 1")
    nonnegative_options = {
        "lambda_foreground_rgb": opt.lambda_foreground_rgb,
        "lambda_full_rgb": opt.lambda_full_rgb,
        "lambda_alpha": opt.lambda_alpha,
        "lambda_center_reg": opt.lambda_center_reg,
        "lambda_flow": opt.lambda_flow,
    }
    for option_name, option_value in nonnegative_options.items():
        if option_value < 0:
            raise ValueError(f"{option_name} must be greater than or equal to zero")
    if opt.ssim_crop_padding < 0:
        raise ValueError("ssim_crop_padding must be greater than or equal to zero")

    if checkpoint:
        # Training checkpoints contain optimizer state and a NumPy scalar in
        # addition to tensors. PyTorch 2.6 defaults torch.load to
        # weights_only=True, which rejects that trusted, locally generated
        # checkpoint structure.
        (model_params, first_iter) = torch.load(
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
                "No matching residual/sweep checkpoint found; initialize angle "
                "corrections from zero"
            )


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    ema_rgbloss_for_log = 0.0
    ema_flowloss_for_log = 0.0
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    flow_idx = 0
    raster_setting_flag = 0

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(0)
        # Use a cyclic successor so every camera is supervised as a current
        # RGB view, including the final training camera in the sequence.
        viewpoint_cam_next = (
            viewpoint_stack[0]
            if viewpoint_stack
            else scene.getTrainCameras()[0]
        )

        use_flow_this_iteration = (
            not args.wo_flow
            and iteration < args.center_axis_lr_max_steps
        )

        if raster_setting_flag == 0:
            fixed_rasterizer = set_rasterizer(viewpoint_cam, gaussians, pipe, bg, scaling_modifier=1.0)
            raster_setting_flag = 1

        gt_image = viewpoint_cam.original_image.cuda()
        H, W = gt_image.shape[-2:]
        if use_flow_this_iteration:
            next_gt_image = viewpoint_cam_next.original_image.cuda()
            h = H // args.flow_downsampling
            w = W // args.flow_downsampling

        # Render
        if not use_flow_this_iteration:
            angle = rotation_angle_for_view(
                viewpoint_cam,
                residual_predictor,
                use_local_residual=not args.wo_tiny,
            )
            axis = gaussians.get_axis(cam_idx)
            center = gaussians.get_center(cam_idx)
            render_pkg = render(fixed_rasterizer, gaussians, axis=axis, center=center, angle=angle)
            image, rendered_alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["rendered_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # use optical flow   
        else: 
            angle1 = rotation_angle_for_view(
                viewpoint_cam,
                residual_predictor,
                use_local_residual=not args.wo_tiny,
            )
            angle2 = rotation_angle_for_view(
                viewpoint_cam_next,
                residual_predictor,
                use_local_residual=not args.wo_tiny,
            ).detach()

            axis = gaussians.get_axis(cam_idx)
            center = gaussians.get_center(cam_idx)
            render_pkg = render(fixed_rasterizer, gaussians, axis=axis, center=center, angle=angle1)
            image, rendered_alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["rendered_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            proj_means_2D, gs_per_pixel, weight_per_gs_pixel = render_pkg["proj_means_2D"], render_pkg["gs_per_pixel"].long(), render_pkg["weight_per_gs_pixel"].detach()

            render_pkg_next = render(fixed_rasterizer, gaussians, axis=axis, center=center, angle=angle2)
            proj_means_2D_next = render_pkg_next["proj_means_2D"].detach()
            
            predicted_flow = compute_flow(proj_means_2D, proj_means_2D_next, gs_per_pixel, weight_per_gs_pixel, w, W, h, H)

        if use_flow_this_iteration:
            # Camera objects persist when viewpoint_stack is replenished, so
            # this lazily builds the cache once and also works after resuming
            # from checkpoints whose absolute iteration is already large.
            if viewpoint_cam.flow_idx is None:
                with torch.no_grad():
                    gt_image_flow = gt_image
                    next_gt_image_flow = next_gt_image
                    threshold = 0.03 
                    diff = torch.mean(torch.abs(gt_image - next_gt_image), dim=0)
                    mask = (diff > threshold).float()

                    flow_2d_gt = flownet(gt_image_flow[None]*255, next_gt_image_flow[None]*255) # return flow_predictions, feat_s, feat_t (540x540)
                    H_flow, W_flow = flow_2d_gt[0].shape[-2:]
                    
                    flow_alpha_mask = viewpoint_cam.alpha_mask
                    flow_alpha_mask = flow_alpha_mask * mask

                    if W_flow == w and H_flow == h:
                        flow_2d_gt = flow_2d_gt[0].squeeze() # (2, h, w)
                    else: 
                        flow_2d_gt = torch.nn.functional.interpolate(flow_2d_gt[0], size=(h, w), mode="bilinear").squeeze()
                        flow_2d_gt[0] *= w / W_flow
                        flow_2d_gt[1] *= h / H_flow # (135x135)
                        flow_alpha_mask = flow_alpha_mask.unsqueeze(0).unsqueeze(0).float()
                        flow_alpha_mask = F.interpolate(flow_alpha_mask, size=(h, w), mode="bilinear")
                        flow_alpha_mask = flow_alpha_mask.squeeze(0).squeeze(0)

                    flow_2d_gt = flow_2d_gt * flow_alpha_mask
                    flow_2d_gt_list.append(flow_2d_gt)
                    flow_alpha_mask_list.append(flow_alpha_mask)
                    viewpoint_cam.flow_idx = flow_idx
                    flow_idx += 1
            else:
                flow_2d_gt = flow_2d_gt_list[viewpoint_cam.flow_idx] 
                flow_alpha_mask = flow_alpha_mask_list[viewpoint_cam.flow_idx]

        # Alpha-aware image losses.  The foreground term is normalized by the
        # foreground area so that white padding cannot dilute product detail.
        foreground_mask = viewpoint_cam.alpha_mask
        foreground_rgb_loss = masked_l1_loss(image, gt_image, foreground_mask)
        full_rgb_loss = l1_loss(image, gt_image)
        image_crop, gt_crop = foreground_crop(
            image,
            gt_image,
            viewpoint_cam.foreground_bbox,
            opt.ssim_crop_padding,
        )
        foreground_ssim_value = ssim(image_crop, gt_crop)
        predicted_alpha = rendered_alpha.squeeze()
        alpha_loss = l1_loss(predicted_alpha, foreground_mask)
        center_reg_loss = gaussians.motion_regularization()

        # flow Loss
        if use_flow_this_iteration:
            if args.wo_UDFS == False:
                weight_udfs = flow_schedule(iteration, max_iter=args.center_axis_lr_max_steps, gamma=args.flow_schedule_lambda, warmup=args.warmup_iteration)
                flow_dir_loss = cosine_flow(predicted_flow, flow_2d_gt.detach().clone(), flow_alpha_mask, w, h)
                flow_mag_dir_loss = L1_flow(predicted_flow, flow_2d_gt.detach().clone(), flow_alpha_mask)
                flow_loss = weight_udfs * flow_dir_loss + (1 - weight_udfs) * flow_mag_dir_loss
            else:
                flow_mag_dir_loss = L1_flow(predicted_flow, flow_2d_gt.detach().clone(), flow_alpha_mask)
                flow_loss = flow_mag_dir_loss

        else:
            flow_loss = image.new_zeros(())

        image_loss = (
            (1.0 - opt.lambda_dssim)
            * opt.lambda_foreground_rgb
            * foreground_rgb_loss
            + opt.lambda_dssim * (1.0 - foreground_ssim_value)
            + opt.lambda_full_rgb * full_rgb_loss
            + opt.lambda_alpha * alpha_loss
        )
        loss = (
            image_loss
            + opt.lambda_flow * flow_loss
            + opt.lambda_center_reg * center_reg_loss
        )
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_rgbloss_for_log = 0.4 * foreground_rgb_loss + 0.6 * ema_rgbloss_for_log
            ema_flowloss_for_log = 0.4 * flow_loss + 0.6 * ema_flowloss_for_log

            if iteration % 10 == 0:
                tilt, side, center_shift = gaussians.motion_summary(cam_idx)
                sweep_deg = torch.rad2deg(
                    residual_predictor.effective_sweep_error(cam_idx)
                )
                progress_bar.set_postfix({
                    "loss": f"{ema_loss_for_log:.4f}",
                    "fg": f"{ema_rgbloss_for_log:.4f}",
                    "alpha": f"{alpha_loss.item():.4f}",
                    "flow": f"{ema_flowloss_for_log:.4f}",
                    "tilt": f"{tilt.item():.2f}",
                    "side": f"{side.item():.2f}",
                    "center": f"{center_shift.item():.4f}",
                    "sweep": f"{sweep_deg.item():.3f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, foreground_rgb_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, gaussians, scene, render,
                            fixed_rasterizer, dataset.train_test_exp, residual_predictor, args.wo_tiny)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                residual_predictor.save_weights(dataset.model_path, iteration)
            
            # 3DGS ADC
            if iteration < opt.densify_until_iter: #15,000

                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) 
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: # 500 / 300
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # 3,000
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                residual_predictor.optimizer.step()
                residual_predictor.update_learning_rate(iteration)
                residual_predictor.optimizer.zero_grad(set_to_none = True)
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                residual_predictor.save_weights(dataset.model_path, iteration)

def prepare_output_and_logger(args, output_name='random', config_args=None):
    if output_name == 'random':  
        if not args.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
    else:
        args.model_path = os.path.join("./output/", output_name)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    if config_args is not None:
        config_args.model_path = args.model_path
        config_to_write = config_args
    else:
        config_to_write = args
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(config_to_write))))

def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, gaussians, scene : Scene, renderFunc, rasterizer, train_test_exp, residual_predictor, wo_tiny):
    # Report test and samples of training set
    if iteration in testing_iterations: #[7000, 30000]
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                foreground_psnr_test = 0.0
                foreground_ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    angle = rotation_angle_for_view(
                        viewpoint,
                        residual_predictor,
                        use_local_residual=not wo_tiny,
                    )
                    axis = gaussians.get_axis(viewpoint.cam_idx)
                    center = gaussians.get_center(viewpoint.cam_idx)

                    image = torch.clamp(renderFunc(rasterizer, scene.gaussians, axis=axis, center=center, angle=angle)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    foreground_psnr_test += masked_psnr(
                        image, gt_image, viewpoint.alpha_mask
                    ).double()
                    image_crop, gt_crop = foreground_crop(
                        image,
                        gt_image,
                        viewpoint.foreground_bbox,
                        args.ssim_crop_padding,
                    )
                    foreground_ssim_test += ssim(image_crop, gt_crop).double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                foreground_psnr_test /= len(config['cameras'])
                foreground_ssim_test /= len(config['cameras'])
                tilt, side, center_shift = gaussians.motion_summary(0)
                sweep_deg = torch.rad2deg(
                    residual_predictor.effective_sweep_error(0)
                )
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} "
                    "FG_PSNR {} FG_SSIM {} | tilt {:.3f} side {:.3f} "
                    "center_shift {:.5f} sweep {:.4f}".format(
                        iteration,
                        config['name'],
                        l1_test,
                        psnr_test,
                        foreground_psnr_test,
                        foreground_ssim_test,
                        tilt.item(),
                        side.item(),
                        center_shift.item(),
                        sweep_deg.item(),
                    )
                )
       
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--fixed_camera", action="store_true", default=True, help="Fix camera position")
    parser.add_argument("--random", action="store_true", default=True, help="Random pointcloud initialization")
    parser.add_argument("--sfm", action="store_true", default=False, help="SfM pointcloud initialization")
    parser.add_argument("--wo_tiny", action="store_true", default=False, help="without tiny angle")
    parser.add_argument(
        "--max_residual_angle_deg",
        type=float,
        default=0.0,
        help="bound residual angle correction to +/- this many degrees; 0 keeps the original unbounded behavior",
    )
    parser.add_argument(
        "--max_sweep_error_deg",
        type=float,
        default=0.0,
        help="learn one linearly accumulated angle correction bounded to +/- this many degrees",
    )
    parser.add_argument("--wo_flow", action="store_true", default=False, help="without optical flow supervision")
    parser.add_argument("--wo_UDFS", action="store_true", default=False, help="without UDFS")
    parser.add_argument(
        "--wo_axis",
        action="store_true",
        default=False,
        help="legacy option: freeze both the rotation axis and rotation center",
    )
    parser.add_argument(
        "--freeze_axis",
        action="store_true",
        default=False,
        help="keep the initialized rotation axis fixed while allowing the center to be learned",
    )
    parser.add_argument(
        "--freeze_center",
        action="store_true",
        default=False,
        help="keep the initialized rotation center fixed while allowing the axis to be learned",
    )
    parser.add_argument(
        "--axis_mode",
        choices=("free", "bounded_tilt"),
        default="free",
        help="free 3D axis or a physically bounded camera/sideways tilt axis",
    )
    parser.add_argument("--axis_tilt_init_deg", type=float, default=30.0)
    parser.add_argument("--axis_tilt_min_deg", type=float, default=0.0)
    parser.add_argument("--axis_tilt_max_deg", type=float, default=90.0)
    parser.add_argument("--axis_side_limit_deg", type=float, default=5.0)
    parser.add_argument("--center_max_offset", type=float, default=0.25)
    parser.add_argument("--center_warmup_iterations", type=int, default=2000)
    parser.add_argument("--multi_camera", action="store_true", default=False, help="multi-camera system")
    parser.add_argument('--name', type=str, default='exper', help='Output folder name')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    if args.sfm == True:
        args.random = False

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")

