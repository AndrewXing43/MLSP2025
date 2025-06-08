import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import os
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from utils import convert_to_polar, convert_to_cartesian, find_FSPL
 
class RadioMapDataset(Dataset):
    def __init__(self, inputs_dir, outputs_dir, sparse_dir, positions_dir, los_dir = None, hit_dir = None, acc_dir = None, freq_filter = None):
        self.inputs_dir = Path(inputs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.sparse_dir = Path(sparse_dir)
        self.positions_dir = Path(positions_dir)
        self.los_dir = Path(los_dir) if los_dir else None
        self.hit_dir = Path(hit_dir) if hit_dir else None
        self.acc_dir = Path(acc_dir) if acc_dir else None

        all_files = [f.name for f in self.inputs_dir.glob("*.png")]
        if freq_filter:
            all_files = [f for f in all_files if f"_f{freq_filter}" in f]

        self.filenames = sorted(all_files)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load RGB (3-channel physical input)
        rgb = Image.open(self.inputs_dir / fname).convert("RGB")
        rgb_tensor = self.to_tensor(rgb)  # [3, H, W]
        C, H, W = rgb_tensor.shape
        center = (W // 2, H // 2)
        rgb_tensor[0] = 255 * rgb_tensor[0] / 20
        # rgb_tensor[1] = 255 * rgb_tensor[1] / 40
        rgb_tensor[2] = 255 * rgb_tensor[2]
        fspl_map = find_FSPL(fname, rgb_tensor[2])
        # fspl_map = fspl_map / fspl_map.max()
        fspl_map = fspl_map.unsqueeze(0)
        rgb_tensor[2] = torch.log10(1 + rgb_tensor[2]) / 2.5
        
        # rgb_tensor[2] = find_FSPL(fname, rgb_tensor[2]) # [H, W] -> [H, W] FSPL map

        # # Running sum on T channel
        # polar_T = convert_to_polar(rgb_tensor[1].unsqueeze(0), center) # [1, H, W] -> [1, num_radial, num_angles]
        # polar_FSPL = convert_to_polar(rgb_tensor[2].unsqueeze(0), center) # [1, H, W] -> [1, num_radial, num_angles]
        # T_cumsum_polar = torch.cumsum(polar_T[0], dim=0)    
        # modified_polar_fspl = polar_FSPL[0] + T_cumsum_polar    

        # modified_fspl_map = convert_to_cartesian(modified_polar_fspl.unsqueeze(0), center, (H, W))
        # rgb_tensor[2] = modified_fspl_map.squeeze(0)

        # # Normalize modified FSPL back to [0, 1]
        # min_val = rgb_tensor[2].min()
        # max_val = rgb_tensor[2].max()
        # rgb_tensor[2] = (rgb_tensor[2] - min_val) / (max_val - min_val + 1e-6)

        # # Delete the original T channel
        # rgb_tensor = torch.stack([rgb_tensor[0], rgb_tensor[2]], dim=0)     
        
        # Load GT PL map (grayscale)
        gt = Image.open(self.outputs_dir / fname).convert("L")
        gt_tensor = self.to_tensor(gt)

        # Load sparse samples (x, y, pl)
        sparse_points = np.load(self.sparse_dir / (Path(fname).stem + ".npy"))
        h, w = gt_tensor.shape[1:]
        sparse_map = torch.zeros((1, h, w))
        mask_map = torch.zeros((1, h, w))
        for x, y, pl in sparse_points:
            sparse_map[0, int(x), int(y)] = pl / 100.0  # Normalization
            mask_map[0, int(x), int(y)] = 1.0
        
        # Load los samples if available
        if self.los_dir:
            los_fname = Path(fname).stem + "_los.npy"
            los_path = self.los_dir / los_fname
            los_map = np.load(los_path)
            los_tensor = torch.from_numpy(los_map).unsqueeze(0) # [1, H, W]
        else:
            los_tensor = torch.zeros((1, h, w))
        
        # Load hit samples if available
        if self.hit_dir:
            hit_fname = Path(fname).stem + "_hit.npy"
            hit_path = self.hit_dir / hit_fname
            hit_map = np.load(hit_path)  # [H, W]
            
            if np.max(hit_map) > 0:
                hit_map = hit_map / np.max(hit_map)
            else:
                hit_map = np.zeros_like(hit_map)

            hit_tensor = torch.from_numpy(hit_map).unsqueeze(0).float()  # [1, H, W]
        else:
            hit_tensor = torch.zeros((1, h, w)).float()

        # Load acc samples if available
        # TODO: Normalization and padding
        if self.acc_dir:
            acc_fname = Path(fname).stem + "_Tsum.npy"
            acc_path = self.acc_dir / acc_fname
            acc_map = np.load(acc_path)  # [H, W]
            acc_tensor = torch.from_numpy(acc_map)
            merged_tensor = acc_tensor + fspl_map
            min_val, max_val = merged_tensor.min(), merged_tensor.max()
            merged_tensor = (merged_tensor - min_val) / (max_val - min_val + 1e-6)
        else:
            merged_tensor = torch.zeros((1, h, w)).float()   
        
        rgb_tensor[1] = merged_tensor

        # Final normalization
        input_tensor = torch.cat([rgb_tensor,sparse_map], dim=0) 
        input_tensor, hit_tensor, gt_tensor, mask_map = self.pad_all(input_tensor, hit_tensor, gt_tensor, mask_map)
        input_tensor = torch.cat([input_tensor, hit_tensor], dim=0)

        return input_tensor, gt_tensor, mask_map

    def pad_all(self, input_tensor, hit_tensor, gt_tensor, mask_tensor):
        _, h, w = input_tensor.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        hit_tensor = F.pad(hit_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        gt_tensor = F.pad(gt_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='constant', value=1)

        return input_tensor, hit_tensor, gt_tensor, mask_tensor

    def _find_position_file(self, fname):
        base = Path(fname).stem  # e.g., B3_Ant3_f1_S2
        building = base.split("_")[0]  # e.g., B3
        s_part = base.split("_")[-1]   # e.g., S2
        s_idx = int(s_part[1:])        # e.g., 2

        pos_file = list(self.positions_dir.glob(f"Positions_{building}_*.csv"))
        if not pos_file:
            raise FileNotFoundError(f"No position file found for {fname}")
        return pos_file[0], s_idx

    def _load_tx_xy(self, filepath, row_index):
        df = pd.read_csv(filepath)
        if row_index >= len(df):
            raise IndexError(f"Tx index {row_index} out of range in {filepath}")
        return float(df.iloc[row_index]['X']), float(df.iloc[row_index]['Y'])
    
    @staticmethod
    def debug_visualize(tensor, title, cmap='viridis'):
        """
        Quick visualization for a (H, W) tensor.
        """
        plt.figure()
        plt.imshow(tensor.cpu().numpy(), cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.axis('off')
        plt.show()