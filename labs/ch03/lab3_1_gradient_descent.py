"""实验 3-1 手算与 NumPy 实现梯度下降

① 用最小二乘公式复算 3.2.3 节 5 套房子的手算结果；② 用 NumPy 从零写梯度下降，比较不同学习率；
③ 在 2400 套房子上比较批量、随机、小批量三种梯度下降。
运行：python lab3_1_gradient_descent.py（在 labs/ch03 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- ① 最小二乘：5 套房子 ----------
x = np.array([50, 70, 90, 110, 130], dtype=float)      # 面积（平方米）
y = np.array([160, 210, 250, 320, 360], dtype=float)   # 总价（万元）
w = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
b = y.mean() - w * x.mean()
pred = w * x + b
print(f"① 最小二乘：w = {w:.2f}，b = {b:.2f}，即 总价 ≈ {w:.2f} × 面积 + {b:.2f}")
print(f"   预测值 {pred.round(1).tolist()}，残差 {(y - pred).round(1).tolist()}，MSE = {((y - pred) ** 2).mean():.1f}")


# ---------- ② 梯度下降 ----------
def gradient_descent(x, y, lr, steps):
    """一元线性回归的批量梯度下降，返回最终的 w、b 和每一步的损失"""
    w, b, history = 0.0, 0.0, []
    n = len(x)
    for _ in range(steps):
        pred = w * x + b
        error = pred - y
        history.append((error ** 2).mean())             # 均方误差
        grad_w = 2 / n * (error * x).sum()               # 损失对 w 的偏导数
        grad_b = 2 / n * error.sum()                     # 损失对 b 的偏导数
        w -= lr * grad_w                                 # 沿梯度的反方向走一步
        b -= lr * grad_b
    return w, b, history


z = (x - x.mean()) / x.std()                             # 面积标准化：均值 90、标准差约 28.28
print(f"\n② 梯度下降（面积先标准化，z = (面积 − {x.mean():.0f}) / {x.std():.2f}）")
print("   学习率   20 步后的损失   200 步后的 w、b        200 步后的损失")
curves = {}
for lr in [0.01, 0.1, 0.5, 0.9, 1.05]:
    wz, bz, hist = gradient_descent(z, y, lr, 200)
    curves[lr] = hist
    print(f"   {lr:<7}{hist[19]:>12.4g}     w={wz:>9.4g}, b={bz:>9.4g}   {hist[-1]:>10.4g}")
wz, bz, _ = gradient_descent(z, y, 0.1, 200)
print(f"   换算回原来的面积：w = {wz / x.std():.2f}，b = {bz - wz * x.mean() / x.std():.2f}（与最小二乘一致）")

# 不标准化，直接用原始面积
for lr in [1e-4, 2e-4]:
    w_raw, b_raw, hist = gradient_descent(x, y, lr, 200)
    print(f"   不标准化，学习率 {lr}：200 步后 w={w_raw:.3g}, b={b_raw:.3g}，损失 {hist[-1]:.3g}")

fig, ax = plt.subplots(figsize=(7, 4))
for lr, hist in curves.items():
    if lr != 1.05:
        ax.plot(hist[:60], label=f"学习率 {lr}")
ax.set(yscale="log", xlabel="迭代次数", ylabel="均方误差（对数刻度）", title="不同学习率的损失下降曲线（前 60 步）")
ax.legend()
plt.tight_layout()
plt.savefig("lab3_1_lr.png", dpi=150)

# ---------- ③ 三种梯度下降：2400 套房子 ----------
df = pd.read_csv("../data/house_clean.csv").sample(2400, random_state=42)
X = ((df["面积"] - df["面积"].mean()) / df["面积"].std()).to_numpy()
Y = df["总价_万元"].to_numpy()
ref = LinearRegression().fit(X.reshape(-1, 1), Y)
best = ((ref.predict(X.reshape(-1, 1)) - Y) ** 2).mean()
print(f"\n③ 2400 套房子，只用面积一个特征。最小二乘的 MSE = {best:.1f}（w={ref.coef_[0]:.2f}，b={ref.intercept_:.2f}）")

rng = np.random.default_rng(0)


def train(batch, lr=0.05, epochs=20):
    w = b = 0.0
    losses, updates = [], 0
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for start in range(0, len(X), batch):
            i = idx[start:start + batch]
            err = w * X[i] + b - Y[i]
            w -= lr * 2 * (err * X[i]).mean()
            b -= lr * 2 * err.mean()
            updates += 1
        losses.append(((w * X + b - Y) ** 2).mean())
    return w, b, losses, updates


print("   方法                   每次用的样本  参数更新次数   第 1 轮后 MSE   第 20 轮后 MSE")
for name, batch in [("批量梯度下降", len(X)), ("小批量梯度下降", 32), ("随机梯度下降", 1)]:
    lr = 0.05 if batch > 1 else 0.005
    w_, b_, losses, updates = train(batch, lr)
    print(f"   {name:<14}{batch:>10}{updates:>12}{losses[0]:>14.1f}{losses[-1]:>14.1f}")
