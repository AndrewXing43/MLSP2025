import os
import numpy as np
from PIL import Image
from pathlib import Path
import math

def generate_uniform_samples(image_path, sampling_rate):
    """
    从路径损耗图像中按采样率均匀采样，返回 [x, y, value] 数组。
    最终采样数量严格等于 ceil(h * w * sampling_rate)。
    """
    img = Image.open(image_path).convert("L")
    pl_array = np.array(img)
    h, w = pl_array.shape
    num_samples = int(np.ceil(sampling_rate * h * w))

    # 估算均匀划分的网格行列数
    aspect_ratio = h / w
    grid_cols = int(np.sqrt(num_samples / aspect_ratio))
    grid_cols = max(1, grid_cols)
    grid_rows = int(np.ceil(num_samples / grid_cols))
    
    # 均匀地生成行列中心坐标
    xs = np.linspace(0, h - 1, grid_rows, dtype=int)
    ys = np.linspace(0, w - 1, grid_cols, dtype=int)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    xx = xx.flatten()
    yy = yy.flatten()

    # 剪裁或随机选择以满足精确数量
    if len(xx) > num_samples:
        indices = np.linspace(0, len(xx) - 1, num_samples, dtype=int)
        xx = xx[indices]
        yy = yy[indices]
    elif len(xx) < num_samples:
        # 补齐
        pad = num_samples - len(xx)
        extra_xx = np.random.randint(0, h, size=pad)
        extra_yy = np.random.randint(0, w, size=pad)
        xx = np.concatenate([xx, extra_xx])
        yy = np.concatenate([yy, extra_yy])

    values = pl_array[xx, yy]
    samples = np.stack([xx, yy, values.astype(np.float32)], axis=1)

    return samples

def process_all_images(outputs_dir, output_dir_05, output_dir_002):
    outputs_dir = Path(outputs_dir)
    output_05_dir = Path(output_dir_05)
    output_002_dir = Path(output_dir_002)

    output_05_dir.mkdir(parents=True, exist_ok=True)
    output_002_dir.mkdir(parents=True, exist_ok=True)

    for img_file in sorted(outputs_dir.glob("*.png")):
        print(f"Processing {img_file.name}...")
        samples_05 = generate_uniform_samples(img_file, sampling_rate=0.005)
        samples_002 = generate_uniform_samples(img_file, sampling_rate=0.0002)

        assert samples_05.shape[0] == math.ceil(0.005 * Image.open(img_file).size[1] * Image.open(img_file).size[0]), "数量不符"
        assert samples_002.shape[0] == math.ceil(0.0002 * Image.open(img_file).size[1] * Image.open(img_file).size[0]), "数量不符"

        np.save(output_05_dir / (img_file.stem + ".npy"), samples_05)
        #np.save(output_002_dir / (img_file.stem + ".npy"), samples_002)

# === 示例调用 ===
import matplotlib.pyplot as plt

def visualize_samples_on_image(image_path, samples, save_path=None, point_size=10):
    """
    在图像上叠加采样点进行可视化。
    - image_path: 原始图像路径
    - samples: [x, y, value] 的 numpy 数组
    - save_path: 保存路径；若为 None，则直接展示
    """
    img = Image.open(image_path).convert("L")
    h, w = img.size[1], img.size[0]
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray', origin='upper')
    
    xs = samples[:, 1]  # 列 → x轴
    ys = samples[:, 0]  # 行 → y轴
    plt.scatter(xs, ys, s=point_size, c='red', marker='o', label='Samples', edgecolors='black')

    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.legend()
        plt.show()

def visualize_from_npy(npy_path, image_dir, save_path=None):
    """
    从 .npy 文件加载采样点，并叠加在对应图像上可视化。
    - npy_path: 采样点的路径，例如 mixed_samples/B1_Ant1_f1_S0.npy
    - image_dir: 原始图像所在目录，例如 outputs/
    """
    npy_path = Path(npy_path)
    image_dir = Path(image_dir)
    samples = np.load(npy_path)
    
    # 从 .npy 文件名恢复图像路径
    image_path = image_dir / (npy_path.stem + ".png")
    
    visualize_samples_on_image(image_path, samples, save_path)


if __name__ == "__main__":
    process_all_images(
        "/home/andrew43/competition/outputs",
        "/home/andrew43/competition/mixed_samples",
        "/home/andrew43/competition/sparse_samples_0.02"
    )
    
    # # 可视化某个样本的采样点
    # visualize_from_npy(
    #     npy_path="mixed_samples/B1_Ant1_f1_S0.npy",
    #     image_dir="outputs",
    #     save_path=None  # 可设为 None 直接显示
    # )



