import numpy as np
import cv2
from PIL import Image
import math
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def harris_corner_candidates(wall_mask, threshold=0.74):
    mask_float = np.float32(wall_mask)
    dst = cv2.cornerHarris(mask_float, blockSize=2, ksize=3, k=0.04)
    dst_norm = cv2.normalize(dst, None, 0, 1, cv2.NORM_MINMAX)
    corner_mask = dst_norm > threshold
    ys, xs = np.where(corner_mask)
    return np.stack([ys, xs], axis=1)  # [N, 2]

def filter_nearest_corners_to_tx(corner_coords, tx_coord, min_dist, max_count):
    dists = np.linalg.norm(corner_coords - np.array(tx_coord), axis=1)
    sorted_indices = np.argsort(dists)
    sorted_coords = corner_coords[sorted_indices]

    selected = []
    for pt in sorted_coords:
        if len(selected) >= max_count:
            break
        if all(np.linalg.norm(pt - sel) >= min_dist for sel in selected):
            selected.append(pt)
    return np.array(selected)

def generate_sample_points(rgb_path, tx_coord, sampling_rate=0.005, corner_ratio=0.2, min_corner_dist=10):
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    H, W = rgb.shape[:2]
    total_samples = math.ceil(H * W * sampling_rate)
    corner_target = max(int(total_samples * corner_ratio), 0)

    # wall_mask = (rgb[:, :, 0] != 0).astype(np.uint8)
    # corners = harris_corner_candidates(wall_mask, threshold=0.74)
    # filtered_corners = filter_nearest_corners_to_tx(corners, tx_coord, min_dist=min_corner_dist, max_count=corner_target)
    # actual_corner_count = len(filtered_corners)

    # 生成均匀采样
    remaining = total_samples
    grid_rows = int(np.sqrt(remaining * H / W))
    grid_cols = int(np.ceil(remaining / grid_rows))
    xs = np.linspace(0, H - 1, grid_rows, dtype=int)
    ys = np.linspace(0, W - 1, grid_cols, dtype=int)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    xx, yy = xx.flatten(), yy.flatten()

    if len(xx) > remaining:
        indices = np.linspace(0, len(xx) - 1, remaining, dtype=int)
        xx, yy = xx[indices], yy[indices]
    elif len(xx) < remaining:
        pad = remaining - len(xx)
        extra_xx = np.random.randint(0, H, size=pad)
        extra_yy = np.random.randint(0, W, size=pad)
        xx = np.concatenate([xx, extra_xx])
        yy = np.concatenate([yy, extra_yy])

    uniform_coords = np.stack([xx, yy], axis=1)
    all_coords = np.concatenate([uniform_coords], axis=0)
    assert all_coords.shape[0] == total_samples, f"采样数不匹配: {all_coords.shape[0]} ≠ {total_samples}"
    return all_coords

def batch_generate(input_dir, output_dir, output_pl_dir, positions_dir,
                   sampling_rate=0.005, corner_ratio=0.2, min_corner_dist=10):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_pl_dir = Path(output_pl_dir)
    positions_dir = Path(positions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(input_dir.glob("*.png"))
    print(f"共检测到 {len(all_images)} 张图像")

    for img_path in tqdm(all_images, desc="生成采样点"):
        try:
            fname = img_path.stem  # e.g., B1_Ant1_f1_S25
            scene = '_'.join(fname.split('_')[:-1])
            s_idx = int(fname.split('_')[-1][1:])  # S25 -> 25
            pos_csv = positions_dir / f"Positions_{scene}.csv"
            df = pd.read_csv(pos_csv, index_col=0)
            tx_x = int(df.loc[s_idx, "X"])
            tx_y = int(df.loc[s_idx, "Y"])
            tx_coord = (tx_x, tx_y)

            coords = generate_sample_points(
                img_path, tx_coord,
                sampling_rate=sampling_rate,
                corner_ratio=corner_ratio,
                min_corner_dist=min_corner_dist
            )

            pl_img = np.array(Image.open(output_pl_dir / img_path.name).convert("L"))
            xs = coords[:, 0].astype(int)
            ys = coords[:, 1].astype(int)
            pl_values = pl_img[xs, ys].astype(np.float32)
            final = np.concatenate([coords, pl_values[:, None]], axis=1)

            save_path = output_dir / (img_path.stem + ".npy")
            np.save(save_path, final)
        except Exception as e:
            print(f"跳过 {img_path.name}，错误：{e}")

if __name__ == "__main__":
    batch_generate(
        input_dir="inputs",            # 原始 RGB 图像目录
        output_dir="sampled",          # 采样点保存目录（将保存为 .npy）
        output_pl_dir="outputs",       # 路径损耗图（灰度图）目录
        positions_dir="Positions",     # TX 坐标 CSV 所在目录
        sampling_rate=0.005,           # 总采样率
        corner_ratio=0,              # 角点占比（其余为均匀采样）
        min_corner_dist=10             # 角点之间最小距离
    )
