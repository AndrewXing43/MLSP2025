import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
import pandas as pd

def visualize_saved_sampling(sample_name, data_root, sparse_dir="task2_sparsified", save=False):
    inputs_dir = Path(data_root) / "inputs"
    positions_dir = Path(data_root) / "Positions"
    sparse_path = Path(data_root) / sparse_dir / sample_name.replace(".png", ".npy")
    vis_dir = Path(data_root) / "sampling_vis"
    vis_dir.mkdir(exist_ok=True)

    # === 加载 RGB 图像 & 生成墙体掩码 ===
    img = Image.open(inputs_dir / sample_name).convert("RGB")
    rgb = transforms.ToTensor()(img)
    R, G = rgb[0].numpy(), rgb[1].numpy()
    wall_mask = ((R != 0) | (G != 0)).astype(np.uint8)
    H, W = wall_mask.shape

    # === 加载采样点 ===
    samples = np.load(sparse_path)  # shape: [N, 3], columns: [x, y, pl]
    coords = [(int(x), int(y)) for x, y, _ in samples]

    # === 稳健地解析场景名和采样索引 ===
    name_no_ext = sample_name.replace(".png", "")
    parts = name_no_ext.split("_")
    scene = "_".join(parts[:-1])
    s_idx = int(parts[-1][1:])  # 去掉 'S'

    # === 读取 TX 坐标 ===
    pos_path = Path(data_root) / "Positions" / f"Positions_{scene}.csv"
    df = pd.read_csv(pos_path, index_col=0)
    tx_x, tx_y = int(df.loc[s_idx, "X"]), int(df.loc[s_idx, "Y"])

    # === 提取墙体坐标 ===
    wall_coords = np.argwhere(wall_mask == 1)
    wall_y, wall_x = wall_coords[:, 1], wall_coords[:, 0]  # note: (row, col) => (x, y)

    # === 绘图 ===
    plt.figure(figsize=(7, 7))
    plt.imshow(np.zeros((H, W)), cmap="gray")  # 黑背景
    plt.scatter(wall_y, wall_x, c="red", s=2, label="Wall Pixels")  # 红色墙体
    plt.scatter([y for x, y in coords], [x for x, y in coords], c="cyan", s=1, label="Sampled Points")
    plt.scatter([tx_y], [tx_x], c="yellow", s=50, marker="*", label="TX Location")
    plt.title(f"Sampling Visualization: {sample_name}")
    plt.axis("off")
    plt.legend(loc="lower right")

    if save:
        out_path = vis_dir / sample_name.replace(".png", "_sampling_overlay.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"✅ Saved to {out_path}")
    else:
        plt.show()

# === 使用示例 ===
visualize_saved_sampling("B1_Ant1_f1_S0.png", data_root="./")
