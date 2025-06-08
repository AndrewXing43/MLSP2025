import numpy as np
from PIL import Image
from pathlib import Path
import math
import os

def generate_priority_uniform_then_wall(pl_image_path, rgb_image_path, sampling_rate=0.005, wall_ratio=0):
    """
    生成采样点：优先均匀采样，再补足墙体点，返回 [x, y, value] 结构，不包含类型标签。
    """
    # === 加载图像 ===
    pl_img = Image.open(pl_image_path).convert("L")
    pl_array = np.array(pl_img)
    h, w = pl_array.shape
    total_samples = int(np.ceil(h * w * sampling_rate))
    uniform_target = int(np.floor(total_samples * (1 - wall_ratio)))
    wall_target = total_samples - uniform_target

    # === 均匀采样 ===
    aspect_ratio = h / w
    grid_cols = int(np.sqrt(uniform_target / aspect_ratio))
    grid_cols = max(1, grid_cols)
    grid_rows = int(np.ceil(uniform_target / grid_cols))
    xs = np.linspace(0, h - 1, grid_rows, dtype=int)
    ys = np.linspace(0, w - 1, grid_cols, dtype=int)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    xx = xx.flatten()
    yy = yy.flatten()

    if len(xx) > uniform_target:
        indices = np.linspace(0, len(xx) - 1, uniform_target, dtype=int)
        xx = xx[indices]
        yy = yy[indices]
    elif len(xx) < uniform_target:
        pad = uniform_target - len(xx)
        extra_xx = np.random.randint(0, h, size=pad)
        extra_yy = np.random.randint(0, w, size=pad)
        xx = np.concatenate([xx, extra_xx])
        yy = np.concatenate([yy, extra_yy])

    uniform_values = pl_array[xx, yy]
    uniform_samples = np.stack([xx, yy, uniform_values.astype(np.float32)], axis=1)

    # === 墙体采样 ===
    rgb_array = np.array(Image.open(rgb_image_path).convert("RGB"))
    wall_mask = (rgb_array[:, :, 0] != 0).astype(np.uint8)
    wall_indices = np.argwhere(wall_mask == 1)

    if len(wall_indices) == 0 or wall_target == 0:
        return uniform_samples

    wall_indices = wall_indices[np.random.choice(len(wall_indices), size=min(wall_target, len(wall_indices)), replace=False)]
    wall_values = pl_array[wall_indices[:, 0], wall_indices[:, 1]]
    wall_samples = np.stack([wall_indices[:, 0], wall_indices[:, 1], wall_values.astype(np.float32)], axis=1)

    # 合并 & 打乱
    all_samples = np.concatenate([uniform_samples, wall_samples], axis=0)
    np.random.shuffle(all_samples)
    return all_samples


def process_all_images_priority_mixed(inputs_dir, outputs_dir, output_dir_mixed, sampling_rate=0.005):
    """
    批量处理所有图像，生成不含类型标签的混合采样 [x, y, value] 格式
    """
    inputs_dir = Path(inputs_dir)
    outputs_dir = Path(outputs_dir)
    output_dir_mixed = Path(output_dir_mixed)
    output_dir_mixed.mkdir(parents=True, exist_ok=True)

    for img_file in sorted(outputs_dir.glob("*.png")):
        print(f"Processing {img_file.name}...")
        rgb_path = inputs_dir / img_file.name
        samples = generate_priority_uniform_then_wall(img_file, rgb_path, sampling_rate=sampling_rate)

        img = Image.open(img_file)
        H, W = img.size[1], img.size[0]
        expected = math.ceil(sampling_rate * H * W)
        assert samples.shape[0] == expected, f"数量不符: got {samples.shape[0]}, expected {expected}"

        np.save(output_dir_mixed / (img_file.stem + ".npy"), samples)


# 示例入口
if __name__ == "__main__":
    process_all_images_priority_mixed(
        inputs_dir="inputs",
        outputs_dir="outputs",
        output_dir_mixed="mixed_samples_05",
        sampling_rate=0.005
    )