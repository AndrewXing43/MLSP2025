import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dataset import RadioMapDataset
from model import UNet
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D  # ✅ 手动添加 legend 句柄
from matplotlib.colors import TwoSlopeNorm

def visualize_error_map(sample_name, model_path, data_root):
    # === 路径设置 ===
    inputs_dir = Path(data_root) / "inputs"
    outputs_dir = Path(data_root) / "outputs"
    sparse_dir = Path(data_root) / "sparse_samples_0.5"
    positions_dir = Path(data_root) / "Positions"
    hit_dir = Path(data_root) / "hitmap"
    acc_dir = Path(data_root) / "Tsummap"

    # === 加载模型 ===
    model = UNet(in_channels=5, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # === 获取样本 ===
    dataset = RadioMapDataset(inputs_dir, outputs_dir, sparse_dir, positions_dir,
                               hit_dir=hit_dir, acc_dir=acc_dir)
    if sample_name not in dataset.filenames:
        raise ValueError(f"Sample {sample_name} not found in dataset.")
    idx = dataset.filenames.index(sample_name)
    input_tensor, gt_tensor, mask_tensor = dataset[idx]

    # === 推理预测 ===
    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0)  # [1, 5, H, W]
        pred_tensor = model(input_tensor).squeeze(0)  # [1, H, W]

    # === 裁剪回原始尺寸 ===
    H_gt = 464
    W_gt = 348
    pred = pred_tensor[0, :H_gt, :W_gt].cpu().numpy() * 255
    gt = gt_tensor[0, :H_gt, :W_gt].cpu().numpy() * 255
    mask = mask_tensor[0, :H_gt, :W_gt].cpu().numpy()
    valid_mask = 1 - mask  # 非采样区域为1，需要监督

    # === Masked RMSE 计算 ===
    error_map = np.abs(pred - gt)
    rmse = np.sqrt(((error_map ** 2) * valid_mask).sum() / max(valid_mask.sum(), 1))
    print(f"Masked RMSE = {rmse:.4f}")

    # === 生成墙体轮廓图（基于 R 通道） ===
    rgb = input_tensor[0, :3, :H_gt, :W_gt].cpu().numpy()
    wall_map = (rgb[0] != 0)

    # === 提取 TX 位置 ===
    sample_base = Path(sample_name).stem
    scene = "_".join(sample_base.split("_")[:-1])
    s_idx = int(sample_base.split("_")[-1][1:])
    pos_path = Path(positions_dir) / f"Positions_{scene}.csv"
    df = pd.read_csv(pos_path, index_col=0)
    tx_x, tx_y = int(df.loc[s_idx, "X"]), int(df.loc[s_idx, "Y"])

    # === 可视化 ===
    fig, ax = plt.subplots(figsize=(6, 6))

    # 显示误差图（不遮蔽采样区域）
    im = ax.imshow(error_map, cmap='afmhot')

    # 色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04, aspect = 22)
    cbar.ax.tick_params(labelsize=15)  # ✅ 控制柱形图旁边数字的大小


    # 墙体轮廓图
    ax.contour(wall_map, levels=[0.5], colors='white', linewidths=0.5)

    # 发射器标记
    tx_handle = ax.scatter(tx_y, tx_x, c='cyan', s=50, marker='x', label='TX')

    # 图例（墙体 + TX）
    wall_handle = Line2D([0], [0], color='white', linewidth=0.5, label='Wall')
    ax.legend(handles=[wall_handle, tx_handle], loc="upper right", fontsize=10, frameon=True, facecolor='gray')

    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample_name = "B1_Ant1_f1_S0.png"
    model_path = "./checkpoints/best_model.pth"
    data_root = "./"
    visualize_error_map(sample_name, model_path, data_root)
