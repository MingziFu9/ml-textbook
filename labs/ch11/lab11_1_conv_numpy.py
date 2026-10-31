"""实验 11-1 手算卷积与池化

① 用 NumPy 实现卷积（步长、填充）和最大池化，在 5×5 的小图上验证手算结果。
② 把一张彩色照片转成灰度，用模糊、锐化、竖边、横边四个卷积核处理，观察特征图；再做 2×2 最大池化。
③ 与 PyTorch 的卷积层比较，确认算的是同一件事；比较全连接层与卷积层的参数个数。
图片：scikit-learn 自带的示例照片 china.jpg（427 × 640，彩色）。
运行：python lab11_1_conv_numpy.py（在 labs/ch11 目录下）
"""
import logging
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_sample_image

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def conv2d(img, k, stride=1, pad=0):
    """单通道卷积：卷积核在图上滑动，每个位置对应元素相乘再相加"""
    if pad:
        img = np.pad(img, pad)                       # 四周补 0
    kh, kw = k.shape
    oh, ow = (img.shape[0] - kh) // stride + 1, (img.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            win = img[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = (win * k).sum()
    return out


def max_pool(img, size=2):
    """最大池化：每个 size × size 的格子里取最大值"""
    h, w = img.shape[0] // size, img.shape[1] // size
    return img[:h * size, :w * size].reshape(h, size, w, size).max(axis=(1, 3))


# ① 小图验证
x = np.array([[0, 0, 1, 1, 1]] * 5, dtype=float)       # 左边暗、右边亮，中间有一条竖直的边
k_v = np.array([[-1, 0, 1]] * 3, dtype=float)          # 竖边检测核：右边减左边
print("① 5×5 小图（左暗右亮）：")
print("\n".join("   " + " ".join(f"{v:.0f}" for v in r) for r in x))
print("  竖边卷积核：\n" + "\n".join("   " + " ".join(f"{v:>2.0f}" for v in r) for r in k_v))
for stride, pad in [(1, 0), (1, 1), (2, 1)]:
    o = conv2d(x, k_v, stride, pad)
    print(f"  步长 {stride}、填充 {pad}：特征图 {o.shape[0]}×{o.shape[1]}，")
    print("\n".join("   " + " ".join(f"{v:>2.0f}" for v in r) for r in o))
print("  输出尺寸公式 (n + 2p − k) / s + 1：5×5、3×3 核 → 不填充 3，填充 1 得 5，填充 1 步长 2 得 3")
f = np.array([[1, 3, 2, 0], [4, 6, 1, 1], [0, 2, 5, 7], [1, 1, 3, 2]], dtype=float)
print(f"  4×4 特征图 {f.astype(int).tolist()} 做 2×2 最大池化 → {max_pool(f).astype(int).tolist()}")

# ② 处理真实照片
rgb = load_sample_image("china.jpg").astype(float) / 255
gray = rgb @ np.array([0.299, 0.587, 0.114])           # 彩色转灰度的常用加权
print(f"\n② 照片：彩色 {rgb.shape}（高 × 宽 × 通道），转为灰度 {gray.shape}，像素值 0～1")
kernels = {
    "模糊（3×3 平均）": np.ones((3, 3)) / 9,
    "锐化": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "竖边（Sobel）": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
    "横边（Sobel）": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
}
maps = {}
t = time.perf_counter()
for name, k in kernels.items():
    maps[name] = conv2d(gray, k)
    m = maps[name]
    print(f"  {name:<10}特征图 {m.shape[0]}×{m.shape[1]}，取值 {m.min():.2f}～{m.max():.2f}，平均绝对值 {np.abs(m).mean():.3f}")
print(f"  用 Python 循环做 4 次卷积用时 {time.perf_counter() - t:.1f} 秒")
edge = np.abs(maps["竖边（Sobel）"])
pooled = max_pool(edge)
print(f"  竖边特征图 {edge.shape[0]}×{edge.shape[1]} → 2×2 最大池化 {pooled.shape[0]}×{pooled.shape[1]}，"
      f"像素数从 {edge.size} 减到 {pooled.size}（四分之一）")

# ③ 与 PyTorch 比较
conv = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False)
with torch.no_grad():
    conv.weight[:] = torch.tensor(kernels["竖边（Sobel）"]).reshape(1, 1, 3, 3)
    t = time.perf_counter()
    out = conv(torch.tensor(gray, dtype=torch.float32).reshape(1, 1, *gray.shape))[0, 0].numpy()
    sec = time.perf_counter() - t
print(f"\n③ PyTorch Conv2d 同一个卷积核：输出 {out.shape[0]}×{out.shape[1]}，与 NumPy 结果的最大差别 "
      f"{np.abs(out - maps['竖边（Sobel）']).max():.1e}，用时 {sec * 1000:.0f} 毫秒")
print("  参数个数比较（输入 28×28 灰度图）：")
print(f"    全连接层，输出 128 个神经元：28×28×128 + 128 = {28 * 28 * 128 + 128}")
print(f"    卷积层，16 个 3×3 卷积核：3×3×1×16 + 16 = {3 * 3 * 1 * 16 + 16}")
print(f"    卷积层，输入 16 通道、输出 32 通道：3×3×16×32 + 32 = {3 * 3 * 16 * 32 + 32}")

fig, ax = plt.subplots(2, 3, figsize=(13, 6.2))
ax[0, 0].imshow(rgb); ax[0, 0].set_title("原图（彩色）")
for a, (name, m) in zip([ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]], maps.items()):
    show = np.abs(m) if "边" in name else np.clip(m, 0, 1)
    a.imshow(show, cmap="gray"); a.set_title(name)
ax[1, 2].imshow(pooled, cmap="gray"); ax[1, 2].set_title(f"竖边特征图 2×2 最大池化（{pooled.shape[0]}×{pooled.shape[1]}）")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.savefig("lab11_1_conv.png", dpi=90)
