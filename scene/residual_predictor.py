import torch
import torch.nn as nn
import math
from utils.general_utils import get_expon_lr_func
import matplotlib.pyplot as plt
from utils.system_utils import searchForMaxIteration
import os
import numpy as np

class ResidualPredictor(nn.Module):
    def __init__(
        self,
        number_of_cameras=1,
        num_ctrl_points=1,
        device="cuda",
        max_residual_angle_deg=0.0,
        max_sweep_error_deg=0.0,
    ):
        super().__init__()
        if max_residual_angle_deg < 0:
            raise ValueError("max_residual_angle_deg must be greater than or equal to 0")
        if max_sweep_error_deg < 0:
            raise ValueError("max_sweep_error_deg must be greater than or equal to 0")

        self.num_ctrl_points = num_ctrl_points
        self.max_residual_angle_rad = (
            math.radians(max_residual_angle_deg)
            if max_residual_angle_deg > 0
            else None
        )
        self.max_sweep_error_rad = math.radians(max_sweep_error_deg)
        self.residuals = nn.Parameter(
            torch.zeros(number_of_cameras, num_ctrl_points + 1, device=device)
        )
        self.sweep_error = nn.Parameter(
            torch.zeros(number_of_cameras, device=device),
            requires_grad=max_sweep_error_deg > 0,
        )
        ctrl_positions = torch.linspace(0, 1, num_ctrl_points + 1, device=device)
        self.register_buffer("ctrl_positions", ctrl_positions)
        self.optimizer = None

    def train_setting(self, training_args):
        l = [
            {'params': self.parameters(),
             'lr': training_args.residual_lr_init,
             "name": "residual"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.rotpredictor_scheduler_args = get_expon_lr_func(lr_init=training_args.residual_lr_init,
                                                       lr_final=training_args.residual_lr_final,
                                                       max_steps=training_args.residual_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "residual":
                lr = self.rotpredictor_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def plot_residual(self, output_folder='.'):
        residuals = np.degrees(
            self._apply_bound(self.residuals[0]).detach().cpu().numpy()
        )
        sweep_error = math.degrees(
            float(self.effective_sweep_error(0).detach().cpu())
        )
        ctrl_positions = self.ctrl_positions.detach().cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(ctrl_positions, -residuals, marker='o', linestyle='-', color='r')
        plt.xlabel('Control Point (Normalized Time)')
        plt.ylabel('Residual Value (degree)')
        plt.title(f'Residuals over Control Points; sweep={sweep_error:.4f} deg')
        plt.savefig(f"{output_folder}/graph/residual.png")
        plt.close()

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "residual_predictor/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'residual_predictor.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "residual_predictor"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "residual_predictor/iteration_{}/residual_predictor.pth".format(loaded_iter))
        state_dict = torch.load(weights_path, map_location=self.residuals.device)
        self.load_state_dict(state_dict, strict=False)

    def effective_sweep_error(self, cam_idx: int):
        raw_error = self.sweep_error[cam_idx]
        if self.max_sweep_error_rad <= 0:
            return raw_error.new_zeros(())
        limit = self.max_sweep_error_rad
        return limit * torch.tanh(raw_error / limit)

    def angle_correction(
        self,
        time: torch.Tensor,
        cam_idx: int,
        use_local_residual: bool,
    ):
        correction = time * self.effective_sweep_error(cam_idx)
        if use_local_residual:
            correction = correction + self.forward(time, cam_idx)
        return correction

    def forward(self, time: torch.Tensor, cam_idx:int):
        segment_idx = torch.bucketize(time, self.ctrl_positions[1:], right=False)

        t0 = self.ctrl_positions[segment_idx]
        t1 = self.ctrl_positions[segment_idx + 1]

        r0 = self.residuals[cam_idx, segment_idx]
        r1 = self.residuals[cam_idx, segment_idx + 1]

        alpha = (time - t0) / (t1 - t0 + 1e-8)
        residual_t = (1 - alpha) * r0 + alpha * r1

        return self._apply_bound(residual_t)

    def _apply_bound(self, residual: torch.Tensor) -> torch.Tensor:
        if self.max_residual_angle_rad is None:
            return residual

        limit = self.max_residual_angle_rad
        return limit * torch.tanh(residual / limit)

    
