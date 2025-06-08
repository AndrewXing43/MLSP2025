import cupy as cp
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

def _bresenhamlines_integer(start, end):
    N = start.shape[0]
    dy = end[:, 0] - start[:, 0]
    dx = end[:, 1] - start[:, 1]
    steps = cp.maximum(cp.abs(dy), cp.abs(dx)) + 1
    max_len = int(cp.max(steps).item())
    t = cp.arange(max_len, dtype=cp.int32).reshape(1, -1)
    t = cp.broadcast_to(t, (N, max_len))
    ratio = cp.clip(t / (steps[:, None] - 1 + 1e-6), 0, 1)
    y = cp.rint(start[:, 0:1] + ratio * dy[:, None]).astype(cp.int32)
    x = cp.rint(start[:, 1:2] + ratio * dx[:, None]).astype(cp.int32)
    lines = cp.stack((y, x), axis=-1)
    return lines

def generate_wall_mask(png_path):
    img = Image.open(png_path).convert("RGB")
    rgb_tensor = transforms.ToTensor()(img)
    R, G = rgb_tensor[0].numpy(), rgb_tensor[1].numpy()
    return ((R != 0) | (G != 0)).astype(np.uint8)

def load_transmission(png_path):
    img = Image.open(png_path).convert("RGB")
    rgb_tensor = transforms.ToTensor()(img)
    G = 255 * rgb_tensor[1].numpy()
    return G

def generate_Tsum_map(transmission, tx_x, tx_y):
    H, W = transmission.shape
    transmission_gpu = cp.asarray(transmission, dtype=cp.float32)
    x, y = cp.meshgrid(cp.arange(H), cp.arange(W), indexing='ij')
    all_points = cp.stack((x.ravel(), y.ravel()), axis=1).astype(cp.int32)
    tx = cp.array([[int(tx_x), int(tx_y)]], dtype=cp.int32)
    tx_batch = cp.repeat(tx, all_points.shape[0], axis=0)

    lines = _bresenhamlines_integer(all_points, tx_batch)
    ys, xs = lines[..., 0], lines[..., 1]
    valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    flat_idx = ys * W + xs
    flat_idx[~valid] = 0
    flat_vals = transmission_gpu.ravel()[flat_idx]
    flat_vals[~valid] = 0
    flat_vals = flat_vals.reshape(all_points.shape[0], -1)

    prev, curr = flat_vals[:, :-1], flat_vals[:, 1:]
    add_mask = (prev == 0) & (curr > 0)
    contrib = cp.zeros_like(curr)
    contrib[add_mask] = curr[add_mask]
    result = cp.sum(contrib, axis=1)
    return cp.asnumpy(result.reshape(H, W))

def process_all(inputs_dir, positions_dir, output_dir):
    inputs_dir = Path(inputs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images = sorted(inputs_dir.glob("*.png"))
    for img_path in tqdm(all_images, desc="Generating Tsum Maps"):
        try:
            fname = img_path.stem
            scene, s_idx = '_'.join(fname.split('_')[:-1]), int(fname.split('_')[-1][1:])
            pos_path = Path(positions_dir) / f"Positions_{scene}.csv"
            df = pd.read_csv(pos_path)
            tx_x = int(df.loc[s_idx, "X"].item())
            tx_y = int(df.loc[s_idx, "Y"].item())
            transmission = load_transmission(img_path)
            Tsum_map = generate_Tsum_map(transmission, tx_x, tx_y)
            np.save(output_dir / f"{scene}_S{s_idx}_Tsum.npy", Tsum_map)
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

def plot_ocm_with_wall_points(npy_path, rgb_path, tx_x=None, tx_y=None, save_path=None):
    rcParams['font.family'] = 'Times New Roman'
    oc_map = np.load(npy_path)
    oc_map = np.clip(oc_map, 0, 255)
    norm = Normalize(vmin=0, vmax=oc_map.max())
    cmap = plt.get_cmap('viridis')
    oc_rgb = cmap(norm(oc_map))[..., :3]

    wall_mask = generate_wall_mask(rgb_path)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.imshow(oc_rgb)

    # 墙体轮廓（白线）
    ax.contour(wall_mask, levels=[0.5], colors='white', linewidths=0.5)
    wall_handle = Line2D([0], [0], color='white', linewidth=0.5, label='Wall')

    # 发射器（TX）点
    handles = [wall_handle]
    if tx_x is not None and tx_y is not None:
        tx_handle = ax.scatter(tx_y, tx_x, color='cyan', s=100, marker='x',
                               linewidths=1.5, label='Tx')
        handles.append(tx_handle)

    # 色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.08, pad=0.04, aspect=14.56)
    cbar.ax.tick_params(labelsize=15)

    # 图标题和图例

    ax.legend(handles=handles, loc='upper left', fontsize=15, frameon=True, facecolor='gray')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ✅ 调用入口示例
if __name__ == "__main__":
    cp.cuda.Device().use()
    # process_all("inputs", "Positions", "Tsummap")  # 如需批量生成
    plot_ocm_with_wall_points(
        npy_path="hitmap/B1_Ant1_f1_S0_hit.npy",
        rgb_path="inputs/B1_Ant1_f1_S0.png",
        tx_x=24,
        tx_y=249,
        save_path=None  # 或者填入 "output_Tsum.png"
    )
