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

import torch
import numpy as np
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, get_cosine_lr_func, build_rotation, axis_angle2rotmat, build_quaternion
from utils.general_utils import cartesian_to_spherical, spherical_to_cartesian, inverse_activate_theta_phi, get_pose_angle, quaternion_multiply, rotate_vector_by_quaternion
from utils.multi_camera_dataset import axis_vectors_from_elevations
# from utils.reloc_utils import compute_relocation_cuda
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from e3nn import o3
from PIL import Image


def estimate_fixed_rotation_center(cam_infos, camera_distance):
    """Back-project the mean foreground centroid into the object-depth plane."""
    if not cam_infos or not np.isfinite(camera_distance) or camera_distance <= 0:
        return None

    world_centers = []
    for cam in cam_infos:
        with Image.open(cam.image_path) as source_image:
            if "A" in source_image.getbands():
                foreground = np.asarray(
                    source_image.getchannel("A"), dtype=np.float64
                ) / 255.0
            else:
                rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
                foreground = np.any(rgb < 250, axis=-1).astype(np.float64)

        foreground_sum = float(foreground.sum())
        if foreground_sum <= 0:
            continue

        yy, xx = np.indices(foreground.shape, dtype=np.float64)
        u = float((foreground * xx).sum() / foreground_sum)
        v = float((foreground * yy).sum() / foreground_sum)
        camera_point = np.array(
            [
                (u - cam.cx) * camera_distance / cam.fx,
                (v - cam.cy) * camera_distance / cam.fy,
                camera_distance,
            ],
            dtype=np.float32,
        )
        # COLMAP convention used here: X_camera = R.T @ X_world + T.
        world_point = np.asarray(cam.R, dtype=np.float32) @ (
            camera_point - np.asarray(cam.T, dtype=np.float32)
        )
        world_centers.append(world_point)

    if not world_centers:
        return None
    return np.mean(world_centers, axis=0, dtype=np.float32)


def estimate_camera_depth_offsets(
    camera_groups, reference_camera_index, camera_distance
):
    """Estimate per-pass depth from median foreground angular width."""
    angular_widths = []
    for cam_infos in camera_groups:
        widths = []
        for cam in cam_infos:
            with Image.open(cam.image_path) as source_image:
                if "A" in source_image.getbands():
                    foreground = np.asarray(source_image.getchannel("A")) > 127
                else:
                    rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
                    foreground = np.any(rgb < 250, axis=-1)
            columns = np.flatnonzero(np.any(foreground, axis=0))
            if columns.size:
                widths.append(float(columns[-1] - columns[0] + 1) / cam.fx)
        angular_widths.append(float(np.median(widths)) if widths else None)

    reference_width = angular_widths[reference_camera_index]
    if reference_width is None or reference_width <= 0:
        return None
    offsets = []
    for angular_width in angular_widths:
        if angular_width is None or angular_width <= 0:
            return None
        depth = camera_distance * reference_width / angular_width
        offsets.append(depth - camera_distance)
    offsets[reference_camera_index] = 0.0
    return np.asarray(offsets, dtype=np.float32)


def _quaternion_align_y_to_axis(axis: torch.Tensor) -> torch.Tensor:
    """Return a differentiable quaternion mapping canonical +Y to ``axis``."""
    target = torch.nn.functional.normalize(axis, p=2, dim=0)
    reference = target.new_tensor([0.0, 1.0, 0.0])
    dot = torch.clamp(torch.dot(reference, target), -1.0, 1.0)
    cross = torch.linalg.cross(reference, target)
    quaternion = torch.cat(((1.0 + dot).reshape(1), cross))
    opposite = target.new_tensor([0.0, 1.0, 0.0, 0.0])
    quaternion = torch.where(dot < -0.999999, opposite, quaternion)
    return torch.nn.functional.normalize(quaternion, p=2, dim=0)


class GaussianModel:
    def __init__(
        self,
        sh_degree,
        optimizer_type="default",
        fixed_camera=False,
        wo_axis=False,
        multi_camera=False,
        number_of_cameras=1,
        freeze_axis=False,
        freeze_center=False,
        freeze_depth=False,
        axis_mode="free",
        axis_tilt_init_deg=30.0,
        axis_tilt_min_deg=0.0,
        axis_tilt_max_deg=90.0,
        axis_side_limit_deg=5.0,
        axis_tilt_deviation_limit_deg=None,
        center_max_offset=0.25,
        center_warmup_iterations=2000,
        depth_max_offset=2.0,
        depth_warmup_iterations=0,
        depth_reference_camera_index=0,
        axis_tilt_init_degrees=None,
        multi_camera_transform="legacy",
    ):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.distance = None
        self.fixed_camera = fixed_camera
        self.wo_axis = wo_axis
        # Keep --wo_axis backward-compatible: historically it froze both the
        # rotation axis and the rotation center.
        self.freeze_axis = wo_axis or freeze_axis
        self.freeze_center = wo_axis or freeze_center
        self.freeze_depth = wo_axis or freeze_depth
        if multi_camera_transform not in {"legacy", "rigid"}:
            raise ValueError("multi_camera_transform must be 'legacy' or 'rigid'")
        self.multi_camera_transform = multi_camera_transform
        if axis_mode not in {"free", "bounded_tilt"}:
            raise ValueError("axis_mode must be either 'free' or 'bounded_tilt'")
        if not 0 <= axis_tilt_min_deg < axis_tilt_max_deg <= 180:
            raise ValueError("axis tilt bounds must satisfy 0 <= min < max <= 180")
        if not axis_tilt_min_deg <= axis_tilt_init_deg <= axis_tilt_max_deg:
            raise ValueError("initial axis tilt must lie inside the configured bounds")
        camera_axis_tilt_init_degrees = None
        if axis_tilt_init_degrees is not None:
            camera_axis_tilt_init_degrees = tuple(
                float(value) for value in axis_tilt_init_degrees
            )
            if len(camera_axis_tilt_init_degrees) != number_of_cameras:
                raise ValueError(
                    "per-camera initial axis tilts must match number_of_cameras"
                )
            if not all(
                math.isfinite(value) for value in camera_axis_tilt_init_degrees
            ):
                raise ValueError("per-camera initial axis tilts must be finite")
            if axis_mode == "bounded_tilt" and not all(
                axis_tilt_min_deg <= value <= axis_tilt_max_deg
                for value in camera_axis_tilt_init_degrees
            ):
                raise ValueError(
                    "per-camera initial axis tilts must lie inside configured bounds"
                )
        if axis_tilt_deviation_limit_deg is not None:
            if (
                not math.isfinite(axis_tilt_deviation_limit_deg)
                or axis_tilt_deviation_limit_deg < 0
            ):
                raise ValueError("axis tilt deviation limit must be finite and nonnegative")
            if camera_axis_tilt_init_degrees is None:
                raise ValueError("axis tilt deviation limit requires per-camera initial tilts")
        if not 0 <= axis_side_limit_deg < 90:
            raise ValueError("axis_side_limit_deg must lie in [0, 90)")
        if center_max_offset < 0:
            raise ValueError("center_max_offset must be greater than or equal to zero")
        if center_warmup_iterations < 0:
            raise ValueError("center_warmup_iterations must be greater than or equal to zero")
        if not math.isfinite(depth_max_offset) or depth_max_offset < 0:
            raise ValueError("depth_max_offset must be finite and nonnegative")
        if depth_warmup_iterations < 0:
            raise ValueError("depth_warmup_iterations must be nonnegative")
        if not 0 <= depth_reference_camera_index < number_of_cameras:
            raise ValueError("depth_reference_camera_index is outside the camera range")
        self.axis_mode = axis_mode
        self.axis_tilt_init_deg = float(axis_tilt_init_deg)
        self.camera_axis_tilt_init_degrees = camera_axis_tilt_init_degrees
        self.axis_tilt_deviation_limit_deg = (
            None
            if axis_tilt_deviation_limit_deg is None
            else float(axis_tilt_deviation_limit_deg)
        )
        self.axis_tilt_min_rad = math.radians(axis_tilt_min_deg)
        self.axis_tilt_max_rad = math.radians(axis_tilt_max_deg)
        self.axis_side_limit_rad = math.radians(axis_side_limit_deg)
        self.center_max_offset = float(center_max_offset)
        self.center_warmup_iterations = int(center_warmup_iterations)
        self.depth_max_offset = float(depth_max_offset)
        self.depth_warmup_iterations = int(depth_warmup_iterations)
        self.depth_reference_camera_index = int(depth_reference_camera_index)
        self.multi_camera = multi_camera
        self.number_of_cameras = number_of_cameras
        if self.camera_axis_tilt_init_degrees is not None:
            self._axis = torch.tensor(
                axis_vectors_from_elevations(
                    self.camera_axis_tilt_init_degrees
                ),
                dtype=torch.float,
            )
        else:
            if self.axis_mode == "bounded_tilt":
                tilt = math.radians(self.axis_tilt_init_deg)
                axis_init = torch.tensor([0.0, math.cos(tilt), math.sin(tilt)])
            else:
                axis_init = torch.tensor([0.0, 1.0, 0.0])
            self._axis = axis_init[None, :].repeat(self.number_of_cameras, 1)
        center_point_init = torch.tensor([0.0, 0.0, 0.0])  # shape (3,)
        self._center_point = center_point_init[None, :].repeat(self.number_of_cameras, 1)  # shape (num_cameras, 3)
        self._center_initial = self._center_point.detach().clone()
        self._camera_depth = torch.zeros(self.number_of_cameras)
        self._camera_depth_initial = self._camera_depth.detach().clone()
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.axis_activation = lambda x: torch.nn.functional.normalize(x, p=2, dim=0)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._axis.detach(),
            self._center_point.detach(),
            self._center_initial.detach(),
            self._camera_depth.detach(),
            self._camera_depth_initial.detach(),
        )
    
    def restore(self, model_args, training_args):
        core_args = model_args[:12]
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = core_args
        if len(model_args) >= 15:
            axis, center_point, center_initial = model_args[12:15]
            self._axis = nn.Parameter(
                axis.detach().clone().requires_grad_(not self.freeze_axis)
            )
            self._center_point = nn.Parameter(
                center_point.detach().clone().requires_grad_(not self.freeze_center)
            )
            self._center_initial = center_initial.detach().clone()
        if len(model_args) >= 17:
            camera_depth, camera_depth_initial = model_args[15:17]
            self._camera_depth = nn.Parameter(
                camera_depth.detach().clone().requires_grad_(not self.freeze_depth)
            )
            self._camera_depth_initial = camera_depth_initial.detach().clone()
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError:
            print(
                "Checkpoint optimizer predates camera-depth parameters; "
                "using freshly initialized optimizer state"
            )
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_exposure(self):
        return self._exposure
   
    def get_axis(self, cam_idx):
        axis = self.axis_activation(self._axis[cam_idx])
        if self.axis_mode == "free":
            return axis

        side = torch.asin(torch.clamp(axis[0], -1.0, 1.0))
        tilt = torch.atan2(axis[2], axis[1])
        side = torch.clamp(
            side, -self.axis_side_limit_rad, self.axis_side_limit_rad
        )
        if self.axis_tilt_deviation_limit_deg is not None:
            initial_tilt = math.radians(
                self.camera_axis_tilt_init_degrees[cam_idx]
            )
            deviation = math.radians(self.axis_tilt_deviation_limit_deg)
            lower_tilt = max(self.axis_tilt_min_rad, initial_tilt - deviation)
            upper_tilt = min(self.axis_tilt_max_rad, initial_tilt + deviation)
            tilt = torch.clamp(tilt, lower_tilt, upper_tilt)
        else:
            tilt = torch.clamp(
                tilt, self.axis_tilt_min_rad, self.axis_tilt_max_rad
            )
        cos_side = torch.cos(side)
        return torch.stack(
            [
                torch.sin(side),
                cos_side * torch.cos(tilt),
                cos_side * torch.sin(tilt),
            ]
        )
    
    def _raw_depth_from_effective(self, effective_depth):
        if self.depth_max_offset == 0:
            return torch.zeros_like(effective_depth)
        normalized = torch.clamp(
            effective_depth / self.depth_max_offset, -0.999999, 0.999999
        )
        return self.depth_max_offset * torch.atanh(normalized)

    def get_camera_depth(self, cam_idx):
        if (
            not self.multi_camera
            or self.multi_camera_transform != "rigid"
            or self.depth_max_offset == 0
            or cam_idx == self.depth_reference_camera_index
        ):
            return self._camera_depth[cam_idx].new_zeros(())
        return self.depth_max_offset * torch.tanh(
            self._camera_depth[cam_idx] / self.depth_max_offset
        )

    def get_lateral_center(self, cam_idx):
        if self.center_max_offset == 0:
            return self._center_initial[cam_idx]
        delta = self._center_point[cam_idx] - self._center_initial[cam_idx]
        if self.multi_camera and self.multi_camera_transform == "rigid":
            delta = delta * delta.new_tensor([1.0, 1.0, 0.0])
        bounded_delta = self.center_max_offset * torch.tanh(
            delta / self.center_max_offset
        )
        return self._center_initial[cam_idx] + bounded_delta

    def get_center(self, cam_idx):
        center = self.get_lateral_center(cam_idx)
        if self.multi_camera and self.multi_camera_transform == "rigid":
            depth_translation = torch.stack(
                [
                    center.new_zeros(()),
                    center.new_zeros(()),
                    self.get_camera_depth(cam_idx),
                ]
            )
            center = center + depth_translation
        return center

    def axis_side_regularization(self):
        """Penalize longitude drift while leaving it learnable."""
        if self.freeze_axis or self.axis_side_limit_rad == 0:
            return self._xyz.new_zeros(())
        raw_axes = torch.nn.functional.normalize(self._axis, p=2, dim=1)
        side = torch.asin(torch.clamp(raw_axes[:, 0], -1.0, 1.0))
        return torch.mean((side / self.axis_side_limit_rad).square())

    def motion_regularization(self):
        if self.freeze_center or self.center_max_offset == 0:
            return self._xyz.new_zeros(())
        center_delta = torch.stack(
            [
                self.get_lateral_center(cam_idx) - self._center_initial[cam_idx]
                for cam_idx in range(self.number_of_cameras)
            ]
        )
        return torch.mean(center_delta.square())

    def depth_regularization(self):
        if self.freeze_depth or self.depth_max_offset == 0:
            return self._xyz.new_zeros(())
        depths = torch.stack(
            [self.get_camera_depth(cam_idx) for cam_idx in range(self.number_of_cameras)]
        )
        return torch.mean(
            ((depths - self._camera_depth_initial) / self.depth_max_offset).square()
        )

    def motion_summary(self, cam_idx=0):
        axis = self.get_axis(cam_idx)
        tilt = torch.rad2deg(torch.atan2(axis[2], axis[1]))
        side = torch.rad2deg(torch.asin(torch.clamp(axis[0], -1.0, 1.0)))
        center_shift = torch.linalg.vector_norm(
            self.get_lateral_center(cam_idx) - self._center_initial[cam_idx]
        )
        return tilt, side, center_shift
   
    def rotate_shs(self, shs_feat, rotation_matrix):
        """Rotate every active spherical-harmonic band, not only degree one."""
        rotation_matrix = rotation_matrix.detach()
        device = rotation_matrix.device
        dtype = rotation_matrix.dtype
        permutation = torch.tensor(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            device=device,
            dtype=dtype,
        )
        permuted = (permutation.T @ rotation_matrix @ permutation).cpu()
        angles = o3.matrix_to_angles(permuted)

        pieces = [shs_feat[:, 0:1, :]]
        offset = 1
        for degree in range(1, self.max_sh_degree + 1):
            width = 2 * degree + 1
            coefficients = shs_feat[:, offset : offset + width, :]
            if degree <= self.active_sh_degree:
                matrix = o3.wigner_D(
                    degree, angles[0], -angles[1], angles[2]
                ).to(device=device, dtype=dtype)
                coefficients = torch.matmul(matrix, coefficients)
            pieces.append(coefficients)
            offset += width
        return torch.cat(pieces, dim=1)
    
    def rotate_gaussian(self, axis, center, angle):  # axis : [ux, uy, uz]
        position = self.get_xyz
        rotation = self.get_rotation
        features = self.get_features

        rotmat = axis_angle2rotmat(axis, angle)
        angle_half = angle / 2
        ux, uy, uz = axis
        q_global = torch.stack([
            torch.cos(angle_half),
            ux * torch.sin(angle_half),
            uy * torch.sin(angle_half),
            uz * torch.sin(angle_half)
        ], dim=0)
        q_global = q_global.repeat(1, position.shape[0]).T

        new_position = position - center
        new_position = rotate_vector_by_quaternion(new_position, q_global)
        new_position = new_position + center

        new_rotation = quaternion_multiply(q_global, rotation)
        new_rotation = torch.nn.functional.normalize(new_rotation, p=2, dim=1)

        new_features = self.rotate_shs(features, rotmat)
        return new_position, new_rotation, new_features
    
    def multi_rotate_gaussian(self, axis, center, angle):
        position = self.get_xyz
        rotation = self.get_rotation
        features = self.get_features

        if self.multi_camera_transform == "legacy":
            pose_angle = get_pose_angle(axis)
            pose_axis = position.new_tensor([1.0, 0.0, 0.0])
            pose_angle_half = pose_angle / 2
            q_pose = torch.stack(
                [
                    torch.cos(pose_angle_half),
                    pose_axis[0] * torch.sin(pose_angle_half),
                    pose_axis[1] * torch.sin(pose_angle_half),
                    pose_axis[2] * torch.sin(pose_angle_half),
                ],
                dim=0,
            ).repeat(1, position.shape[0]).T
            rotmat_pose = axis_angle2rotmat(pose_axis, pose_angle)
        else:
            # One physical turntable pivot is the canonical origin.  Each pass
            # has its own camera projection, so its camera-space transform is
            # X_cam_i = R_axis_i R_turntable(theta_i(t)) X_canonical + t_i.
            # ``center`` is t_i: the same pivot expressed in camera i coordinates.
            q_pose_one = _quaternion_align_y_to_axis(axis)
            q_pose = q_pose_one[None, :].repeat(position.shape[0], 1)
            rotmat_pose = build_rotation(q_pose_one[None, :])[0]

        rotate_axis = position.new_tensor([0.0, 1.0, 0.0])
        rotate_angle_half = angle.squeeze() / 2
        q_rot_one = torch.stack(
            [
                torch.cos(rotate_angle_half),
                rotate_angle_half.new_zeros(()),
                torch.sin(rotate_angle_half),
                rotate_angle_half.new_zeros(()),
            ]
        )
        q_rot = q_rot_one[None, :].repeat(position.shape[0], 1)
        rotmat_rot = axis_angle2rotmat(rotate_axis, angle)

        q_total = quaternion_multiply(q_pose, q_rot)
        q_total = torch.nn.functional.normalize(q_total, p=2, dim=1)
        rotmat_total = torch.matmul(rotmat_pose, rotmat_rot)

        new_position = rotate_vector_by_quaternion(position, q_total)
        if self.multi_camera_transform == "rigid":
            new_position = new_position + center
        new_rotation = quaternion_multiply(q_total, rotation)
        new_rotation = torch.nn.functional.normalize(new_rotation, p=2, dim=1)
        new_features = self.rotate_shs(features, rotmat_total)

        return new_position, new_rotation, new_features

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        canonical_points = np.asarray(pcd.points, dtype=np.float32)
        point_cloud_center = np.mean(canonical_points, axis=0, dtype=np.float32)
        center_source = "point-cloud mean"
        depth_offsets_np = np.zeros(self.number_of_cameras, dtype=np.float32)
        center_points_np = np.repeat(
            point_cloud_center[None, :], self.number_of_cameras, axis=0
        )
        if self.fixed_camera and self.multi_camera and self.multi_camera_transform == "rigid":
            grouped_infos = [
                [cam for cam in cam_infos if cam.cam_idx == camera_index]
                for camera_index in range(self.number_of_cameras)
            ]
            estimated_depth_offsets = estimate_camera_depth_offsets(
                grouped_infos,
                self.depth_reference_camera_index,
                self.distance,
            )
            if estimated_depth_offsets is not None:
                if self.depth_max_offset > 0:
                    depth_offsets_np = np.clip(
                        estimated_depth_offsets,
                        -0.95 * self.depth_max_offset,
                        0.95 * self.depth_max_offset,
                    ).astype(np.float32)
                depth_for_center = self.distance + depth_offsets_np
            else:
                depth_for_center = np.full(
                    self.number_of_cameras, self.distance, dtype=np.float32
                )
            estimated_centers = [
                estimate_fixed_rotation_center(group, depth_for_center[index])
                for index, group in enumerate(grouped_infos)
            ]
            if all(center is not None for center in estimated_centers):
                center_points_np = np.stack(estimated_centers).astype(np.float32)
                center_points_np[:, 2] = 0.0
                canonical_points = canonical_points - point_cloud_center
                center_source = "per-camera alpha-mask centroids"
        elif self.fixed_camera and not self.multi_camera:
            image_center = estimate_fixed_rotation_center(cam_infos, self.distance)
            if image_center is not None:
                center_points_np[0] = image_center
                center_source = "alpha-mask centroid"
        center_point = torch.tensor(center_points_np).float().cuda()
        depth_initial = torch.tensor(
            depth_offsets_np, dtype=torch.float, device="cuda"
        )
        if self.camera_axis_tilt_init_degrees is not None:
            axis = torch.tensor(
                axis_vectors_from_elevations(
                    self.camera_axis_tilt_init_degrees
                ),
                dtype=torch.float,
                device="cuda",
            )
        elif self.multi_camera: # Camera numbers increase from front view to top view
            if self.number_of_cameras == 7: # axis initialization for multi-camera system
                axis = torch.tensor([[0,0.8,0.2],[0,0.8,0.2],[0,0.8,0.2],[0,0.71,0.71],[0,0.2,0.8],[0,0.2,0.8],[0,0.2,0.8]]).float().cuda()
            elif self.number_of_cameras == 6:
                axis = torch.tensor([[0,0.8,0.2],[0,0.8,0.2],[0,0.71,0.71],[0,0.71,0.71],[0,0.2,0.8],[0,0.2,0.8]]).float().cuda()
            elif self.number_of_cameras == 5:
                axis = torch.tensor([[0,0.8,0.2],[0,0.8,0.2],[0,0.7,0.7],[0,0.2,0.8],[0,0.2,0.8]]).float().cuda()
            elif self.number_of_cameras == 4:
                axis = torch.tensor([[0,0.8,0.2],[0,0.8,0.2],[0,0.2,0.8],[0,0.2,0.8]]).float().cuda()
            elif self.number_of_cameras == 3:
                axis = torch.tensor([[0,0.8,0.2],[0,0.7,0.7],[0,0.8,0.2]]).float().cuda()
            else:  
                axis = torch.tensor([0, 0.7, 0.7]).float().cuda()
                axis = axis[None, :].repeat(self.number_of_cameras, 1) # shape (num_cameras, 3)
        else:
            axis = torch.tensor([[0, 1.0, 0.0]]).float().cuda() # initial axis(camera up vector), (single_camera)

        if (
            self.axis_mode == "bounded_tilt"
            and self.camera_axis_tilt_init_degrees is None
        ):
            tilt = math.radians(self.axis_tilt_init_deg)
            bounded_axis = torch.tensor(
                [0.0, math.cos(tilt), math.sin(tilt)],
                dtype=torch.float,
                device="cuda",
            )
            axis = bounded_axis[None, :].repeat(self.number_of_cameras, 1)
        
        print(f"initial axis: {axis}")
        print(f"initial lateral center ({center_source}): {center_point}")
        print(f"initial camera depth offsets: {depth_initial}")

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(canonical_points).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._center_point = nn.Parameter(
            center_point, requires_grad=not self.freeze_center
        )
        self._center_initial = center_point.detach().clone()
        self._axis = nn.Parameter(axis, requires_grad=not self.freeze_axis)
        self._camera_depth_initial = depth_initial.detach().clone()
        self._camera_depth = nn.Parameter(
            self._raw_depth_from_effective(depth_initial),
            requires_grad=not self.freeze_depth,
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense #0.01
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._axis], 'lr': training_args.center_axis_lr_init, "name": "axis"}, 
            {'params': [self._center_point], 'lr': training_args.center_axis_lr_init, "name": "center_point"}, 
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.multi_camera and self.multi_camera_transform == "rigid":
            l.insert(
                3,
                {
                    "params": [self._camera_depth],
                    "lr": training_args.depth_lr_init,
                    "name": "camera_depth",
                },
            )

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
        self.axis_scheduler_args = get_cosine_lr_func(training_args.center_axis_lr_init, training_args.center_axis_lr_final,
                                                        max_steps=training_args.center_axis_lr_max_steps)
        self.bounded_axis_scheduler_args = get_cosine_lr_func(
            training_args.axis_lr_init,
            training_args.axis_lr_final,
            max_steps=training_args.center_axis_lr_max_steps,
        )
        self.center_scheduler_args = get_cosine_lr_func(
            training_args.center_lr_init,
            training_args.center_lr_final,
            max_steps=max(
                1,
                training_args.center_axis_lr_max_steps
                - self.center_warmup_iterations,
            ),
        )
        self.depth_scheduler_args = get_cosine_lr_func(
            training_args.depth_lr_init,
            training_args.depth_lr_final,
            max_steps=max(
                1,
                training_args.depth_lr_max_steps
                - self.depth_warmup_iterations,
            ),
        )
  

    def update_learning_rate(self, iteration, geometry_iteration=None, pose_iteration=None):
        ''' Learning rate scheduling per step '''
        geometry_iteration = iteration if geometry_iteration is None else geometry_iteration
        pose_iteration = iteration if pose_iteration is None else pose_iteration
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(geometry_iteration)
                param_group['lr'] = lr
                
            if param_group['name'] == "axis":
                scheduler = (
                    self.bounded_axis_scheduler_args
                    if self.axis_mode == "bounded_tilt"
                    else self.axis_scheduler_args
                )
                lr = 0.0 if self.freeze_axis else scheduler(pose_iteration)
                param_group['lr'] = lr

            if param_group['name'] == "center_point":
                if self.freeze_center or pose_iteration <= self.center_warmup_iterations:
                    lr = 0.0
                else:
                    lr = self.center_scheduler_args(
                        pose_iteration - self.center_warmup_iterations
                    )
                param_group["lr"] = lr

            if param_group["name"] == "camera_depth":
                if self.freeze_depth or pose_iteration <= self.depth_warmup_iterations:
                    lr = 0.0
                else:
                    lr = self.depth_scheduler_args(
                        pose_iteration - self.depth_warmup_iterations
                    )
                param_group["lr"] = lr

    def _clear_optimizer_group_gradients(self, group_names):
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") in group_names:
                for parameter in param_group["params"]:
                    parameter.grad = None

    def clear_geometry_gradients(self):
        """Freeze all canonical-cloud parameters during pose calibration."""
        self._clear_optimizer_group_gradients(
            {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
        )

    def clear_structure_gradients(self):
        """Freeze shape/topology while allowing SH appearance to adapt."""
        self._clear_optimizer_group_gradients(
            {"xyz", "opacity", "scaling", "rotation"}
        )

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        axis = torch.stack(
            [self.get_axis(i) for i in range(self.number_of_cameras)]
        ).detach().cpu().numpy()
        center = torch.stack(
            [self.get_center(i) for i in range(self.number_of_cameras)]
        ).detach().cpu().numpy()
        lateral_center = torch.stack(
            [self.get_lateral_center(i) for i in range(self.number_of_cameras)]
        ).detach().cpu().numpy()
        camera_depth = torch.stack(
            [self.get_camera_depth(i) for i in range(self.number_of_cameras)]
        ).detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        axis_path = os.path.join(os.path.dirname(path), "axis.npy")
        np.save(axis_path, axis)
        center_path = os.path.join(os.path.dirname(path), "center.npy")
        saved_center = (
            lateral_center
            if self.multi_camera and self.multi_camera_transform == "rigid"
            else center
        )
        np.save(center_path, saved_center)
        if self.multi_camera and self.multi_camera_transform == "rigid":
            depth_path = os.path.join(os.path.dirname(path), "camera_depth.npy")
            np.save(depth_path, camera_depth)
        motion_path = os.path.join(os.path.dirname(path), "motion.json")
        motion_data = {
            "axis_mode": self.axis_mode,
            "multi_camera_transform": self.multi_camera_transform,
            "axis": axis.tolist(),
            "camera_axis_tilt_initial_degrees": self.camera_axis_tilt_init_degrees,
            "axis_tilt_deviation_limit_degrees": self.axis_tilt_deviation_limit_deg,
            "center": center.tolist(),
            "lateral_center": lateral_center.tolist(),
            "center_initial": self._center_initial.detach().cpu().numpy().tolist(),
            "camera_depth": camera_depth.tolist(),
            "camera_depth_initial": (
                self._camera_depth_initial.detach().cpu().numpy().tolist()
            ),
        }
        with open(motion_path, "w", encoding="utf-8") as motion_file:
            json.dump(motion_data, motion_file, indent=2)

    def reset_scale_rotation(self):
        dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)
        scales_new = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rotation_new = torch.zeros((self.get_xyz.shape[0], 4), device="cuda")
        rotation_new[:, 0] = 1
        optimizable_tensors_scaling = self.replace_tensor_to_optimizer(scales_new, "scaling")
        optimizable_tensors_rot = self.replace_tensor_to_optimizer(rotation_new, "rotation")
        self._scaling = optimizable_tensors_scaling["scaling"]
        self._rotation = optimizable_tensors_rot["rotation"]

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)

        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        axis_path = os.path.join(os.path.dirname(path), "axis.npy")
        if os.path.exists(axis_path):
            axis = np.load(axis_path)
            self._axis = nn.Parameter(
                torch.tensor(axis, dtype=torch.float, device="cuda").requires_grad_(
                    not self.freeze_axis
                )
            )

        center_path = os.path.join(os.path.dirname(path), "center.npy")
        if os.path.exists(center_path):
            center_point = np.load(center_path)
            center_tensor = torch.tensor(
                center_point, dtype=torch.float, device="cuda"
            )
            self._center_point = nn.Parameter(
                center_tensor.requires_grad_(not self.freeze_center)
            )
            self._center_initial = center_tensor.detach().clone()

        if self.multi_camera and self.multi_camera_transform == "rigid":
            depth_path = os.path.join(os.path.dirname(path), "camera_depth.npy")
            if os.path.exists(depth_path):
                depth_values = np.load(depth_path)
            else:
                depth_values = np.zeros(self.number_of_cameras, dtype=np.float32)
            depth_tensor = torch.tensor(
                depth_values, dtype=torch.float, device="cuda"
            )
            self._camera_depth_initial = depth_tensor.detach().clone()
            self._camera_depth = nn.Parameter(
                self._raw_depth_from_effective(depth_tensor),
                requires_grad=not self.freeze_depth,
            )


    def load_geometry_from_ply(
        self, path, *, canonicalize_for_rigid_multi_camera=True, source_camera_index=0
    ):
        """Load Gaussian geometry while preserving this run's camera parameters."""
        camera_axis = self._axis.detach().clone().cuda()
        camera_center = self._center_point.detach().clone().cuda()
        camera_center_initial = self._center_initial.detach().clone().cuda()
        camera_depth = self._camera_depth.detach().clone().cuda()
        camera_depth_initial = self._camera_depth_initial.detach().clone().cuda()

        self.load_ply(path)
        # Scene construction sized this buffer for its temporary random cloud;
        # imported geometry can contain a completely different point count.
        self.max_radii2D = torch.zeros(
            self.get_xyz.shape[0], dtype=torch.float, device="cuda"
        )
        source_axis = self._axis.detach().clone()
        source_center = self._center_point.detach().clone()
        if not 0 <= source_camera_index < source_axis.shape[0]:
            raise ValueError("source_camera_index is outside the source model range")

        if canonicalize_for_rigid_multi_camera:
            axis = source_axis[source_camera_index]
            center = source_center[source_camera_index]
            pose = _quaternion_align_y_to_axis(axis)
            inverse_pose = pose.clone()
            inverse_pose[1:] = -inverse_pose[1:]
            quaternion = inverse_pose[None, :].repeat(self.get_xyz.shape[0], 1)
            rotation_matrix = build_rotation(inverse_pose[None, :])[0]

            xyz = rotate_vector_by_quaternion(self.get_xyz - center, quaternion)
            rotations = quaternion_multiply(quaternion, self.get_rotation)
            rotations = torch.nn.functional.normalize(rotations, p=2, dim=1)
            features = self.rotate_shs(self.get_features, rotation_matrix)
            self._xyz = nn.Parameter(xyz.detach().requires_grad_(True))
            self._rotation = nn.Parameter(rotations.detach().requires_grad_(True))
            self._features_dc = nn.Parameter(
                features[:, :1, :].detach().requires_grad_(True)
            )
            self._features_rest = nn.Parameter(
                features[:, 1:, :].detach().requires_grad_(True)
            )

        self._axis = nn.Parameter(
            camera_axis.requires_grad_(not self.freeze_axis)
        )
        self._center_point = nn.Parameter(
            camera_center.requires_grad_(not self.freeze_center)
        )
        self._center_initial = camera_center_initial
        self._camera_depth = nn.Parameter(
            camera_depth.requires_grad_(not self.freeze_depth)
        )
        self._camera_depth_initial = camera_depth_initial
        print(
            f"Initialized canonical geometry from {path} "
            f"({self.get_xyz.shape[0]} Gaussians)"
        )

    def load_motion_from_json(self, path):
        """Load calibrated camera motion while retaining the current geometry."""
        with open(path, "r", encoding="utf-8") as motion_file:
            motion = json.load(motion_file)

        saved_transform = motion.get("multi_camera_transform")
        if saved_transform != self.multi_camera_transform:
            raise ValueError(
                "motion transform mismatch: "
                f"saved={saved_transform!r}, current={self.multi_camera_transform!r}"
            )

        device = self._axis.device
        axis = torch.as_tensor(motion["axis"], dtype=torch.float32, device=device)
        lateral_values = motion.get("lateral_center")
        if lateral_values is None:
            lateral_values = motion["center"]
        lateral_center = torch.as_tensor(
            lateral_values,
            dtype=torch.float32,
            device=device,
        )
        camera_depth = torch.as_tensor(
            motion.get("camera_depth", [0.0] * self.number_of_cameras),
            dtype=torch.float32,
            device=device,
        )
        expected_vector_shape = (self.number_of_cameras, 3)
        if tuple(axis.shape) != expected_vector_shape:
            raise ValueError(
                f"motion axis shape {tuple(axis.shape)} does not match "
                f"{expected_vector_shape}"
            )
        if tuple(lateral_center.shape) != expected_vector_shape:
            raise ValueError(
                f"motion center shape {tuple(lateral_center.shape)} does not match "
                f"{expected_vector_shape}"
            )
        if tuple(camera_depth.shape) != (self.number_of_cameras,):
            raise ValueError(
                f"motion depth shape {tuple(camera_depth.shape)} does not match "
                f"({self.number_of_cameras},)"
            )
        if not all(
            torch.isfinite(value).all()
            for value in (axis, lateral_center, camera_depth)
        ):
            raise ValueError("motion initializer contains non-finite values")

        # Re-center the regularization priors on the accepted calibration. This
        # preserves the exact imported pose and, when left learnable, permits
        # only bounded refinement around that pose rather than the old metadata
        # initialization.
        if self.multi_camera_transform == "rigid":
            lateral_center = lateral_center.clone()
            lateral_center[:, 2] = 0.0
        self._axis = nn.Parameter(
            axis.clone(), requires_grad=not self.freeze_axis
        )
        self._center_initial = lateral_center.detach().clone()
        self._center_point = nn.Parameter(
            lateral_center.clone(), requires_grad=not self.freeze_center
        )
        self._camera_depth_initial = camera_depth.detach().clone()
        self._camera_depth = nn.Parameter(
            self._raw_depth_from_effective(camera_depth),
            requires_grad=not self.freeze_depth,
        )
        print(f"Initialized calibrated camera motion from {path}")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in {"axis", "center_point", "camera_depth"}:
                continue 
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] in {"axis", "center_point", "camera_depth"}:
                    continue 
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] in {"axis", "center_point", "camera_depth"}:
                continue 
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    @staticmethod
    def _limit_densification_mask(selected_mask, scores, max_selected):
        """Keep the highest-gradient selections within an optional point budget."""
        if max_selected is None:
            return selected_mask
        max_selected = max(0, int(max_selected))
        selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
        if selected_indices.numel() <= max_selected:
            return selected_mask
        if max_selected == 0:
            return torch.zeros_like(selected_mask)
        selected_scores = scores.flatten()[selected_indices]
        keep = selected_indices[
            torch.topk(selected_scores, k=max_selected, sorted=False).indices
        ]
        limited_mask = torch.zeros_like(selected_mask)
        limited_mask[keep] = True
        return limited_mask

    def densify_and_split(
        self, grads, grad_threshold, scene_extent, N=2, max_new_points=None
    ):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        max_selected = (
            None
            if max_new_points is None
            else int(max_new_points) // N
        )
        selected_pts_mask = self._limit_densification_mask(
            selected_pts_mask, padded_grad, max_selected
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(
        self, grads, grad_threshold, scene_extent, max_new_points=None
    ):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        selected_pts_mask = self._limit_densification_mask(
            selected_pts_mask, torch.norm(grads, dim=-1), max_new_points
        )
        
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        radii,
        max_gaussians=0,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if self.fixed_camera:
            extent = self.distance

        point_limit = int(max_gaussians) if max_gaussians > 0 else None
        self.tmp_radii = radii
        clone_budget = (
            None
            if point_limit is None
            else max(0, point_limit - self.get_xyz.shape[0])
        )
        self.densify_and_clone(
            grads, max_grad, extent, max_new_points=clone_budget
        )
        split_budget = (
            None
            if point_limit is None
            else max(0, point_limit - self.get_xyz.shape[0])
        )
        self.densify_and_split(
            grads, max_grad, extent, max_new_points=split_budget
        )

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
        self.prune_points(prune_mask)
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
