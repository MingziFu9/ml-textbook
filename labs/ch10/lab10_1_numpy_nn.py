"""实验 10-1 用 NumPy 从零实现神经网络

① 感知机学“与”“或”“异或”：前两个能学会，异或学不会。
② 两层神经网络（2 个输入、4 个隐藏神经元、1 个输出）学异或：手写前向传播和反向传播，用梯度下降训练。
③ 用数值方法检查反向传播算出的梯度是否正确。
④ 画出两层网络学到的决策边界，以及隐藏层把四个点“搬”到了哪里。
运行：python lab10_1_numpy_nn.py（在 labs/ch10 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
GATES = {"与": np.array([0, 0, 0, 1]), "或": np.array([0, 1, 1, 1]), "异或": np.array([0, 1, 1, 0])}

# ① 感知机：w ← w + η(y − ŷ)x，b ← b + η(y − ŷ)
print("① 感知机（初始 w = [0, 0]，b = 0，学习率 0.1，最多 20 轮）")
for name, y in GATES.items():
    w, b, errs = np.array([0.0, 0.0]), 0.0, []
    for epoch in range(20):
        e = 0
        for xi, yi in zip(X, y):
            pred = int(w @ xi + b > 0)
            if pred != yi:
                # 保留 10 位小数，避免 0.1 + 0.2 ≠ 0.3 这类浮点误差影响“是否大于 0”的判断
                w, b, e = np.round(w + 0.1 * (yi - pred) * xi, 10), round(b + 0.1 * (yi - pred), 10), e + 1
        errs.append(e)
        if e == 0:
            break
    print(f"  {name}：每轮错误数 {errs}" + (f" → 第 {len(errs)} 轮全对，w = [{w[0]:.2f}, {w[1]:.2f}]，b = {b:.2f}"
                                       if errs[-1] == 0 else " → 20 轮后仍然有错，学不会"))


# ② 两层神经网络
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(params, X):
    W1, b1, W2, b2 = params
    z1 = X @ W1 + b1          # 隐藏层的加权和，形状 (4, 4)
    a1 = sigmoid(z1)          # 隐藏层的输出
    z2 = a1 @ W2 + b2         # 输出层的加权和，形状 (4, 1)
    a2 = sigmoid(z2)          # 网络的输出：判为 1 的概率
    return z1, a1, z2, a2


def loss_fn(a2, y):
    return -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))      # 交叉熵


def backward(params, X, y):
    """反向传播：从输出层往回，一层层算出损失对每个参数的梯度"""
    W1, b1, W2, b2 = params
    z1, a1, z2, a2 = forward(params, X)
    n = len(X)
    d2 = (a2 - y) / n                     # 输出层的“误差”（Sigmoid + 交叉熵时恰好是 预测 − 真实）
    gW2, gb2 = a1.T @ d2, d2.sum(0)
    d1 = (d2 @ W2.T) * a1 * (1 - a1)      # 误差传回隐藏层，再乘 Sigmoid 的导数 a(1 − a)
    gW1, gb1 = X.T @ d1, d1.sum(0)
    return [gW1, gb1, gW2, gb2]


y = GATES["异或"].reshape(-1, 1).astype(float)
rng = np.random.default_rng(0)
params = [rng.normal(0, 1, (2, 4)), np.zeros(4), rng.normal(0, 1, (4, 1)), np.zeros(1)]
lr, history = 1.0, []
print("\n② 两层神经网络学异或（2 → 4 → 1，Sigmoid，交叉熵损失，学习率 1.0，全批量梯度下降）")
print(f"  训练前的输出：{np.round(forward(params, X)[3].ravel(), 3)}，损失 {loss_fn(forward(params, X)[3], y):.3f}")
for epoch in range(1, 5001):
    grads = backward(params, X, y)
    params = [p - lr * g for p, g in zip(params, grads)]
    history.append(loss_fn(forward(params, X)[3], y))
    if epoch in (1, 100, 500, 1000, 2000, 5000):
        out = forward(params, X)[3].ravel()
        print(f"  第 {epoch:>4} 轮：损失 {history[-1]:.4f}，输出 {np.round(out, 3)}，判对 {int(((out > 0.5) == y.ravel()).sum())} 个")
_, a1, _, a2 = forward(params, X)
print("  训练后隐藏层 4 个神经元对四个输入的输出：")
for xi, h in zip(X.astype(int), a1):
    print(f"    ({xi[0]}, {xi[1]}) → " + "，".join(f"{v:.2f}" for v in h))

# ③ 梯度检查
rng = np.random.default_rng(1)
p0 = [rng.normal(0, 1, (2, 4)), rng.normal(0, 0.1, 4), rng.normal(0, 1, (4, 1)), rng.normal(0, 0.1, 1)]
g_bp = backward(p0, X, y)
eps, diffs = 1e-5, []
for k in range(4):
    for idx in np.ndindex(p0[k].shape):
        plus = [p.copy() for p in p0]; plus[k][idx] += eps
        minus = [p.copy() for p in p0]; minus[k][idx] -= eps
        num = (loss_fn(forward(plus, X)[3], y) - loss_fn(forward(minus, X)[3], y)) / (2 * eps)
        diffs.append(abs(num - g_bp[k][idx]))
print(f"\n③ 梯度检查：17 个参数，反向传播与数值差分的最大差别 {max(diffs):.1e}（小于 1e-7 说明反向传播写对了）")
print(f"  例：W2 第 1 个元素的梯度，反向传播 {g_bp[2][0, 0]:.6f}")

# ④ 图
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].plot(history); ax[0].set_xlabel("训练轮数"); ax[0].set_ylabel("交叉熵损失"); ax[0].set_title("损失下降曲线"); ax[0].grid(alpha=0.3)
gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
prob = forward(params, np.c_[gx.ravel(), gy.ravel()])[3].reshape(gx.shape)
ax[1].contourf(gx, gy, prob, levels=20, cmap="RdBu_r", alpha=0.7); ax[1].contour(gx, gy, prob, levels=[0.5], colors="k")
ax[1].scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu_r", edgecolors="k", s=120)
ax[1].set_title("两层网络学到的决策边界（黑线 = 0.5）")
z_out = (a1 @ params[2] + params[3]).ravel()     # 输出神经元对 4 个隐藏输出的加权和
names = [f"({a}, {b})" for a, b in X.astype(int)]
ax[2].barh(names, z_out, color=["#c62828" if t else "#1565c0" for t in y.ravel()])
ax[2].axvline(0, color="k"); ax[2].invert_yaxis()
ax[2].set_xlabel("输出神经元的加权和 z（> 0 判为 1）")
ax[2].set_title("隐藏层输出经过加权求和：一个阈值就能分开")
plt.tight_layout(); plt.savefig("lab10_1_xor.png", dpi=110)
