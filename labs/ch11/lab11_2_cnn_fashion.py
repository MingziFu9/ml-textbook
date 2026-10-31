"""实验 11-2 CNN 服装图片分类

Fashion-MNIST：28×28 灰度服装图片，10 类。为了在普通电脑的 CPU 上几分钟内跑完，只取 12000 张训练
（其中 2000 张做验证），测试用完整的 10000 张。
① 多层感知机（把图片拉平）作对照；② PyTorch 卷积神经网络；③ 同一个 CNN 用 Keras 3 再写一遍；
④ 画训练曲线、混淆最多的类别、认错的图片和第一层卷积的特征图。
第一次运行会自动下载数据（约 30 MB，存在用户目录的 .keras/datasets 下）。
运行：python lab11_2_cnn_fashion.py（在 labs/ch11 目录下），约 2 分钟
"""
import logging
import os
import time

os.environ["KERAS_BACKEND"] = "torch"
import keras                                    # noqa: E402
import matplotlib                               # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402
import torch                                    # noqa: E402
from sklearn.metrics import confusion_matrix    # noqa: E402
from torch import nn                            # noqa: E402

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
torch.set_num_threads(4)

NAMES = ["T恤", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
(x_all, y_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
rng = np.random.default_rng(0)
idx = rng.permutation(len(x_all))[:12000]
x_tr, y_tr = x_all[idx[:10000]], y_all[idx[:10000]]
x_val, y_val = x_all[idx[10000:]], y_all[idx[10000:]]
prep = lambda a: (a.astype("float32") / 255)[:, None, :, :]     # noqa: E731  (张数, 通道 1, 高 28, 宽 28)
Xtr, Xval, Xte = map(lambda a: torch.tensor(prep(a)), (x_tr, x_val, x_test))
ytr, yval, yte = map(lambda a: torch.tensor(a.astype("int64")), (y_tr, y_val, y_test))
print(f"训练 {len(x_tr)} 张，验证 {len(x_val)} 张，测试 {len(x_test)} 张；每张 {x_tr.shape[1]}×{x_tr.shape[2]} 灰度图；"
      f"张量形状 {tuple(Xtr.shape)}")
print("类别：" + "、".join(f"{i} {n}" for i, n in enumerate(NAMES)))
EPOCHS, BATCH = 10, 64
loss_fn = nn.CrossEntropyLoss()


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def fit(model, name):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {"train": [], "val": []}
    t = time.perf_counter()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            b = perm[i:i + BATCH]
            loss = loss_fn(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        hist["train"].append(accuracy(model, Xtr, ytr)); hist["val"].append(accuracy(model, Xval, yval))
    sec = time.perf_counter() - t
    n = sum(p.numel() for p in model.parameters())
    te = accuracy(model, Xte, yte)
    print(f"  {name}：参数 {n}，第 1 轮验证 {hist['val'][0]:.3f}，第 10 轮训练 {hist['train'][-1]:.3f} / 验证 {hist['val'][-1]:.3f}，"
          f"测试 {te:.3f}，用时 {sec:.0f} 秒")
    return hist, te


# ① 多层感知机
print("\n① 多层感知机（784 → 128 → 10）")
torch.manual_seed(0)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
h_mlp, te_mlp = fit(mlp, "MLP")

# ② 卷积神经网络
print("\n② 卷积神经网络（PyTorch）")
torch.manual_seed(0)
cnn = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 16 × 14 × 14
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 × 7 × 7
    nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
)
x = Xtr[:1]
print("  每一层输出的形状：")
for layer in cnn:
    x = layer(x)
    n = sum(p.numel() for p in layer.parameters())
    print(f"    {layer.__class__.__name__:<10}→ {tuple(x.shape[1:])}" + (f"，参数 {n}" if n else ""))
h_cnn, te_cnn = fit(cnn, "CNN")

cnn.eval()
with torch.no_grad():
    pred = cnn(Xte).argmax(1).numpy()
cm = confusion_matrix(y_test, pred)
print("  各类测试准确率：" + "，".join(f"{NAMES[i]} {cm[i, i] / cm[i].sum():.2f}" for i in range(10)))
off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted(((off[i, j], i, j) for i in range(10) for j in range(10) if off[i, j]), reverse=True)[:4]
print("  最常认错：" + "；".join(f"{NAMES[i]}→{NAMES[j]} {c} 张" for c, i, j in pairs))

# ③ Keras 3
print("\n③ 同一个 CNN 用 Keras 3")
keras.utils.set_random_seed(0)
km = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),                    # Keras 默认通道在最后：高 × 宽 × 通道
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"), keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"), keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(10, activation="softmax"),
])
km.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
kx = lambda a: (a.astype("float32") / 255)[..., None]          # noqa: E731
t = time.perf_counter()
kh = km.fit(kx(x_tr), y_tr, epochs=EPOCHS, batch_size=BATCH, validation_data=(kx(x_val), y_val), verbose=0)
sec = time.perf_counter() - t
te_k = km.evaluate(kx(x_test), y_test, verbose=0)[1]
print(f"  Keras CNN：参数 {km.count_params()}，第 10 轮验证 {kh.history['val_accuracy'][-1]:.3f}，测试 {te_k:.3f}，用时 {sec:.0f} 秒")

print("\n④ 汇总（测试集 10000 张）")
print(f"  MLP（PyTorch）  {te_mlp:.3f}")
print(f"  CNN（PyTorch）  {te_cnn:.3f}")
print(f"  CNN（Keras 3）  {te_k:.3f}")

# 画图：训练曲线、认错的图片、第一层特征图
fig = plt.figure(figsize=(13, 6.5))
gs = fig.add_gridspec(4, 6)
a = fig.add_subplot(gs[0:2, 0:2])
for h, n in [(h_mlp, "MLP"), (h_cnn, "CNN")]:
    a.plot(range(1, EPOCHS + 1), h["train"], "--", label=f"{n} 训练")
    a.plot(range(1, EPOCHS + 1), h["val"], label=f"{n} 验证")
a.set_xlabel("轮数"); a.set_ylabel("准确率"); a.set_title("训练曲线"); a.legend(fontsize=8); a.grid(alpha=0.3)
wrong = np.where(pred != y_test)[0][:8]
for k, i in enumerate(wrong):                       # 认错的图片：真实 → 预测
    a = fig.add_subplot(gs[k // 4, 2 + k % 4])
    a.imshow(x_test[i], cmap="gray_r"); a.axis("off")
    a.set_title(f"{NAMES[y_test[i]]}→{NAMES[pred[i]]}", fontsize=9)
with torch.no_grad():
    fm = cnn[1](cnn[0](Xte[:1]))[0].numpy()          # 第一层卷积 + ReLU 后的 16 张特征图
a = fig.add_subplot(gs[2:4, 0:2]); a.imshow(x_test[0], cmap="gray_r"); a.set_title(f"输入：{NAMES[y_test[0]]}"); a.axis("off")
for k in range(8):                                   # 前 8 张特征图
    a = fig.add_subplot(gs[2 + k // 4, 2 + k % 4])
    a.imshow(fm[k], cmap="gray"); a.axis("off"); a.set_title(f"第一层特征图 {k + 1}", fontsize=9)
plt.tight_layout(); plt.savefig("lab11_2_cnn.png", dpi=90)
