import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import math
import cupy as cp
 
def RMSELoss():
     def masked_rmse(pred, target, mask):
         diff = ((pred - target)*255) ** 2
         masked_diff = diff * (1 - mask)  # Only penalize on non-sampled locations
         mse = masked_diff.sum() / (1 - mask).sum().clamp(min=1.0)
         return torch.sqrt(mse)
     return masked_rmse

@torch.no_grad()
def compute_rmse(pred, target, mask):
    diff = ((pred - target) * 255) ** 2
    masked_diff = diff * (1 - mask)
    mse_val = masked_diff.sum() / (1 - mask).sum().clamp(min=1.0)
    rmse_val = torch.sqrt(mse_val)
    return rmse_val
 
def save_checkpoint(model, path):
     os.makedirs(os.path.dirname(path), exist_ok=True)
     torch.save(model.state_dict(), path)
 
def plot_loss_curve(train_losses, val_losses, out_path):
     os.makedirs(os.path.dirname(out_path), exist_ok=True)
     plt.figure()
     plt.plot(train_losses, label='Train Loss')
     plt.plot(val_losses, label='Val Loss')
     plt.xlabel('Epoch')
     plt.ylabel('RMSE Loss')
     plt.title('Training vs Validation Loss')
     plt.legend()
     plt.grid(True)
     plt.savefig(out_path)
     plt.close()
 
def custom_collate_fn(batch):
     # batch: List of (input, target, mask)
     max_h = max(item[0].shape[1] for item in batch)
     max_w = max(item[0].shape[2] for item in batch)
 
     padded_batch = []
     for input_tensor, gt_tensor, mask_tensor in batch:
         pad_h = max_h - input_tensor.shape[1]
         pad_w = max_w - input_tensor.shape[2]
 
         input_p = F.pad(input_tensor, (0, pad_w, 0, pad_h), value=0)
         gt_p = F.pad(gt_tensor, (0, pad_w, 0, pad_h), value=0)
         mask_p = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=1)
 
         padded_batch.append((input_p, gt_p, mask_p))
 
     inputs, gts, masks = zip(*padded_batch)
     return torch.stack(inputs), torch.stack(gts), torch.stack(masks)

def find_FSPL(fname, d_map):
    """
    Args:
        fname (str): filename containing _f1, _f2, _f3 to identify frequency
        d_map (torch.Tensor): (H, W) tensor, real distance in meters
    Returns:
        fspl_map (torch.Tensor): (H, W) tensor, FSPL in dB
    """
    freqs_GHz = [0.868, 1.8, 3.5]
    match = re.search(r'_f(\d+)', fname)
    fnum = int(match.group(1))
    freq_GHz = freqs_GHz[fnum-1]
    lam = 0.3 / freq_GHz
    fspl = (4 * math.pi * d_map) / lam
    fspl = 20 * torch.log10(fspl + 1)
    return fspl

def test_FSPL(d_map):
    freq_GHz = 0.868
    lam = 0.3 / freq_GHz
    fspl = (4 * math.pi * d_map) / lam
    fspl = 20 * torch.log10(fspl + 1)
    return fspl

def convert_to_polar(tensor: torch.Tensor, center: tuple[int, int], num_radial=None, num_angles=None):
     """
     Converts a (C, H, W) tensor to polar coordinates per channel.

     Args:
          tensor (torch.Tensor): Input tensor of shape (C, H, W)
          center (tuple[int, int]): The (x, y) center of the polar transform
          num_radial (int, optional): Number of radial steps (defaults to half of min(H, W))
          num_angles (int, optional): Number of angular steps (defaults to 360)

     Returns:
          torch.Tensor: Polar-transformed tensor of shape (C, num_radial, num_angles)
     """
     C, H, W = tensor.shape
     cx, cy = center

     if num_radial is None:
          num_radial = min(H, W)
     if num_angles is None:
          num_angles = 360

     # Create polar coordinate grid (r, θ)
     max_x = max(cx, W - cx)
     max_y = max(cy, H - cy)
     max_r = int( (max_x**2 + max_y**2) ** 0.5 )

     r = torch.linspace(0, 1, steps=num_radial) * max_r
     theta = torch.linspace(0, 2 * torch.pi, steps=num_angles)
     grid_r, grid_theta = torch.meshgrid(r, theta, indexing='ij')

     # Convert polar to cartesian (normalized to [-1, 1] for grid_sample)
     x = grid_r * torch.cos(grid_theta) + cx
     y = grid_r * torch.sin(grid_theta) + cy

     x_norm = 2 * x / (W - 1) - 1
     y_norm = 2 * y / (H - 1) - 1

     grid = torch.stack((x_norm, y_norm), dim=-1)  # (num_radial, num_angles, 2)

     # Reshape tensor to (1, C, H, W) and grid to (1, num_radial, num_angles, 2)
     tensor = tensor.unsqueeze(0)
     grid = grid.unsqueeze(0)

     # grid_sample expects (N, C, H, W) and grid in (N, H_out, W_out, 2)
     polar_tensor = F.grid_sample(tensor, grid, mode='bilinear', align_corners=True)

     return polar_tensor.squeeze(0).permute(0, 2, 1)

def convert_to_cartesian(polar_tensor: torch.Tensor, center: tuple[int, int], original_size: tuple[int, int]):
    """
    Converts a (C, num_radial, num_angles) polar tensor back to cartesian (C, H, W).
    """
    C, num_radial, num_angles = polar_tensor.shape
    H, W = original_size
    cx, cy = center

    # Step 1: Create (x, y) grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=polar_tensor.device),
        torch.linspace(0, W - 1, W, device=polar_tensor.device),
        indexing='ij'
    )

    # Step 2: Shift to center
    x_shifted = grid_x - cx
    y_shifted = grid_y - cy

    # Step 3: (x, y) -> (r, theta)
    r = torch.sqrt(x_shifted ** 2 + y_shifted ** 2)
    theta = torch.atan2(y_shifted, x_shifted)  # no flip!

    # Step 4: Normalize (r, theta) to [0, 1]
    max_x = max(cx, W - cx)
    max_y = max(cy, H - cy)
    max_r = (max_x**2 + max_y**2) ** 0.5

    r_norm = r / max_r
    theta_norm = (theta + torch.pi) / (2 * torch.pi)

    # Step 5: Map (r_norm, theta_norm) to polar tensor index
    r_idx = r_norm * (num_radial - 1)
    theta_idx = theta_norm * (num_angles - 1)

    # Step 6: Normalize index to [-1, 1] for grid_sample
    r_grid = 2 * r_idx / (num_radial - 1) - 1
    theta_grid = 2 * theta_idx / (num_angles - 1) - 1

    grid = torch.stack((r_grid, theta_grid), dim=-1) 

    grid = grid.unsqueeze(0)  # batch dim
    polar_tensor = polar_tensor.unsqueeze(0)

    # Step 7: Sample
    cartesian_tensor = F.grid_sample(polar_tensor, grid, mode='bilinear', align_corners=True)
    cartesian_tensor = torch.flip(cartesian_tensor, dims=[2,3]) 

    return cartesian_tensor.squeeze(0)