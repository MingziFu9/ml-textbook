"""实验 10-3 训练技巧对比实验

在手写数字数据上比较：① 学习率；② 优化器（SGD、带动量的 SGD、Adam）；③ 训练数据很少时的 Dropout 和早停；
④ 批归一化；⑤ 梯度消失：10 层 Sigmoid 网络与 10 层 ReLU 网络第一层的梯度大小。
运行：python lab10_3_training_tricks.py（在 labs/ch10 目录下），约 1 分钟
"""
import copy
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
torch.set_num_threads(4)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
T = lambda a: torch.tensor(a)                                   # noqa: E731
Xtr, ytr, Xv, yv, Xte, yte = T(X_tr), T(y_tr), T(X_val), T(y_val), T(X_test), T(y_test)
loss_fn = nn.CrossEntropyLoss()


def mlp(hidden=64, layers=1, dropout=0.0, bn=False, act=nn.ReLU):
    mods, d = [], 64
    for _ in range(layers):
        mods += [nn.Linear(d, hidden)] + ([nn.BatchNorm1d(hidden)] if bn else []) + [act()]
        mods += [nn.Dropout(dropout)] if dropout else []
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, 10))


def acc(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def train(model, opt, Xa, ya, epochs, batch=32, patience=None, seed=0):
    torch.manual_seed(seed)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model(Xa[idx]), ya[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            hist["train_loss"].append(loss_fn(model(Xa), ya).item())
            vl = loss_fn(model(Xv), yv).item()
        hist["val_loss"].append(vl); hist["val_acc"].append(acc(model, Xv, yv))
        if patience is not None:
            if vl < best - 1e-4:
                best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    model.load_state_dict(best_state)
                    break
    return hist


# ① 学习率（SGD，20 轮）
print("① 学习率（普通 SGD，20 轮）")
lr_hist = {}
for lr in [0.001, 0.01, 0.1, 1.0, 5.0]:
    torch.manual_seed(0); m = mlp()
    h = train(m, torch.optim.SGD(m.parameters(), lr=lr), Xtr, ytr, 20)
    lr_hist[lr] = h["train_loss"]
    print(f"  学习率 {lr:<6} 第 1 轮训练损失 {h['train_loss'][0]:.3f}，第 20 轮 {h['train_loss'][-1]:.3f}，验证准确率 {h['val_acc'][-1]:.3f}")

# ② 优化器（学习率 0.01，20 轮）
print("\n② 优化器（学习率 0.01，20 轮）")
opt_hist = {}
for name in ["SGD", "SGD + 动量 0.9", "Adam"]:
    torch.manual_seed(0); m = mlp()
    opt = {"SGD": lambda p: torch.optim.SGD(p, lr=0.01),
           "SGD + 动量 0.9": lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
           "Adam": lambda p: torch.optim.Adam(p, lr=0.01)}[name](m.parameters())
    h = train(m, opt, Xtr, ytr, 20)
    opt_hist[name] = h["train_loss"]
    first = next((i + 1 for i, a in enumerate(h["val_acc"]) if a >= 0.95), None)
    print(f"  {name:<14}第 5 轮训练损失 {h['train_loss'][4]:.3f}，第 20 轮 {h['train_loss'][-1]:.3f}，验证准确率 {h['val_acc'][-1]:.3f}，"
          f"验证准确率首次达到 0.95：{'第 ' + str(first) + ' 轮' if first else '未达到'}")

# ③ 训练数据很少时：Dropout 与早停（每个数字只取 20 张训练，大网络 64 → 256 → 256 → 10）
rng = np.random.default_rng(0)
small = np.concatenate([rng.choice(np.where(y_tr == d)[0], 20, replace=False) for d in range(10)])
Xs, ys = Xtr[small], ytr[small]
print(f"\n③ 训练数据很少（{len(small)} 张）、网络很大（两个 256 的隐藏层），Adam 学习率 0.001")
reg_hist = {}
for name, dropout, patience, epochs in [("不加 Dropout，训练 300 轮", 0.0, None, 300),
                                        ("Dropout 0.5，训练 300 轮", 0.5, None, 300),
                                        ("不加 Dropout，早停（耐心 20 轮）", 0.0, 20, 300)]:
    torch.manual_seed(0); m = mlp(hidden=256, layers=2, dropout=dropout)
    h = train(m, torch.optim.Adam(m.parameters(), lr=1e-3), Xs, ys, epochs, batch=16, patience=patience)
    reg_hist[name] = h
    print(f"  {name:<24}实际训练 {len(h['val_loss']):>3} 轮，训练准确率 {acc(m, Xs, ys):.3f}，"
          f"最低验证损失 {min(h['val_loss']):.3f}（第 {int(np.argmin(h['val_loss'])) + 1} 轮），停止时验证损失 {h['val_loss'][-1]:.3f}，"
          f"测试准确率 {acc(m, Xte, yte):.3f}")

# ④ 批归一化（4 个隐藏层，SGD 学习率 0.01，20 轮）
print("\n④ 批归一化（4 个隐藏层，每层 64，普通 SGD 学习率 0.01，20 轮）")
for bn in [False, True]:
    torch.manual_seed(0); m = mlp(layers=4, bn=bn)
    h = train(m, torch.optim.SGD(m.parameters(), lr=0.01), Xtr, ytr, 20)
    print(f"  {'加' if bn else '不加'}批归一化：第 5 轮训练损失 {h['train_loss'][4]:.3f}，第 20 轮 {h['train_loss'][-1]:.3f}，测试准确率 {acc(m, Xte, yte):.3f}")

# ⑤ 梯度消失
print("\n⑤ 梯度消失：10 个隐藏层的网络，未训练时各层权重梯度的大小（平均绝对值）")
for name, act, he in [("Sigmoid", nn.Sigmoid, False), ("ReLU", nn.ReLU, False), ("ReLU + He 初始化", nn.ReLU, True)]:
    torch.manual_seed(0); m = mlp(layers=10, act=act)
    if he:                                     # He 初始化：按输入个数调整初始权重的大小，适合 ReLU
        for mod in m:
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu"); nn.init.zeros_(mod.bias)
    loss_fn(m(Xtr), ytr).backward()
    g = [mod.weight.grad.abs().mean().item() for mod in m if isinstance(mod, nn.Linear)]
    print(f"  {name:<14}第 1 层 {g[0]:.2e}，第 5 层 {g[4]:.2e}，第 10 层 {g[9]:.2e}，输出层 {g[10]:.2e}；第 1 层 ÷ 输出层 = {g[0] / g[10]:.1e}")
print(f"  Sigmoid 导数最大 0.25：连乘 10 层 0.25^10 = {0.25 ** 10:.1e}")

# 画图
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for lr, l in lr_hist.items():
    ax[0].plot(range(1, 21), l, label=f"学习率 {lr}")
ax[0].set_yscale("log"); ax[0].set_title("① 学习率（SGD）"); ax[0].set_xlabel("轮数"); ax[0].set_ylabel("训练损失"); ax[0].legend(fontsize=8)
for n, l in opt_hist.items():
    ax[1].plot(range(1, 21), l, label=n)
ax[1].set_yscale("log"); ax[1].set_title("② 优化器（学习率 0.01）"); ax[1].set_xlabel("轮数"); ax[1].legend(fontsize=8)
for n, h in reg_hist.items():
    ax[2].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], label=n)
ax[2].set_title("③ 验证损失：Dropout 与早停"); ax[2].set_xlabel("轮数"); ax[2].legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("lab10_3_tricks.png", dpi=110)
