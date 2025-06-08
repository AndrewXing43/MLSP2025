import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from PIL import Image
from torchvision import transforms

def generate_wall_mask(png_path):
    """
    从 RGB 图像生成墙体掩码（墙体区域为1）。
    """
    img = Image.open(png_path).convert("RGB")
    rgb_tensor = transforms.ToTensor()(img)
    R, G = rgb_tensor[0].numpy(), rgb_tensor[1].numpy()
    return ((R != 0) | (G != 0)).astype(np.uint8)

def plot_ocm_with_wall_points(npy_path, rgb_path, tx_x=None, tx_y=None, save_path=None):
    """
    绘制 Obstruction Count Map，并标注墙体与 TX 位置。
    
    Args:
        npy_path: .npy 格式的 HitMap 文件路径
        rgb_path: RGB 图像路径，用于生成墙体轮廓
        tx_x, tx_y: 发射器位置（图像中坐标）
        save_path: 可选，若提供则保存图像
    """
    rcParams['font.family'] = 'Times New Roman'

    # === 加载 hitmap 并裁剪最大值 ===
    oc_map = np.load(npy_path)
    oc_map = np.clip(oc_map, 0, 20)

    # === 设置 colormap 和归一化 ===
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=20)
    oc_rgb = cmap(norm(oc_map))[..., :3]

    # === 生成墙体掩码 ===
    wall_mask = generate_wall_mask(rgb_path)

    # === 开始绘图 ===
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.imshow(oc_rgb)

    # 墙体轮廓（白色）
    ax.contour(wall_mask, levels=[0.5], colors='white', linewidths=0.5)
    handles = [Line2D([0], [0], color='white', linewidth=0.5, label='Wall')]

    # 发射器点
    if tx_x is not None and tx_y is not None:
        tx_handle = ax.scatter(tx_y, tx_x, color='cyan', s=100, marker='x',
                               linewidths=1.5, label='Tx')
        handles.append(tx_handle)

    # 图例
    ax.legend(handles=handles, loc='upper left', fontsize=12, frameon=True, facecolor='gray')

    # Colorbar 设置
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.1, pad=0.04, aspect=15)
    cbar.set_label("Obstruction Count", fontsize=12)

    # 标题和布局
    ax.set_title("Obstruction Count Map", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ✅ 调用函数示例
plot_ocm_with_wall_points(
    npy_path="hitmap/B1_Ant1_f1_S0_hit.npy",
    rgb_path="inputs/B1_Ant1_f1_S0.png",
    tx_x=24,
    tx_y=249,
    save_path=None  # 或替换为 "output.png"
)
