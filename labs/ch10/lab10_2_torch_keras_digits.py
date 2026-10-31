"""实验 10-2 PyTorch 与 Keras 识别手写数字

同一个多层感知机（64 → 64 → 10），先用 PyTorch 写一遍，再用 Keras 3 写一遍，比较代码、准确率和用时。
数据：scikit-learn 自带的手写数字（1797 张 8×8 图片），与第 6 章实验 6-3 的划分相同。
运行：python lab10_2_torch_keras_digits.py（在 labs/ch10 目录下）
"""
import os
import time

os.environ["KERAS_BACKEND"] = "torch"          # Keras 3 使用 PyTorch 作为底层，不必再装 TensorFlow
import keras                                   # noqa: E402
import numpy as np                             # noqa: E402
import torch                                   # noqa: E402
from sklearn.datasets import load_digits       # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch import nn                           # noqa: E402

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32")               # 像素 0～16 缩放到 0～1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
print(f"训练 {len(X_tr)} 张，验证 {len(X_val)} 张，测试 {len(X_test)} 张；每张 {X.shape[1]} 个像素，10 个类别")
EPOCHS, BATCH = 30, 32

# ---------- PyTorch ----------
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
n_params = sum(p.numel() for p in model.parameters())
print(f"\n① PyTorch：{model}")
print(f"  参数个数 {n_params}（64×64 + 64 + 64×10 + 10）")
loss_fn = nn.CrossEntropyLoss()                # 内部先做 Softmax，再算交叉熵
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr)
Xv, yv = torch.tensor(X_val), torch.tensor(y_val)
t = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), BATCH):
        idx = perm[i:i + BATCH]
        loss = loss_fn(model(Xt[idx]), yt[idx])   # 前向传播，算损失
        opt.zero_grad()                           # 清空上一步的梯度
        loss.backward()                           # 反向传播，自动求出所有梯度
        opt.step()                                # 按梯度更新参数
    if epoch in (1, 5, 10, 20, 30):
        model.eval()
        with torch.no_grad():
            tl = loss_fn(model(Xt), yt).item()
            va = (model(Xv).argmax(1) == yv).float().mean().item()
        print(f"  第 {epoch:>2} 轮：训练损失 {tl:.4f}，验证准确率 {va:.3f}")
t_torch = time.perf_counter() - t
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_test))
    acc_torch = (logits.argmax(1).numpy() == y_test).mean()
    prob = torch.softmax(logits[:1], dim=1).numpy().ravel()
print(f"  测试准确率 {acc_torch:.3f}，训练用时 {t_torch:.1f} 秒")
print(f"  测试集第 1 张（真实数字 {y_test[0]}）的 Softmax 输出：" + " ".join(f"{p:.3f}" for p in prob)
      + f" → 预测 {prob.argmax()}")

# 张量与自动求导的小例子
w = torch.tensor(2.0, requires_grad=True)
L = (3 * w - 4) ** 2
L.backward()
print(f"  自动求导：L = (3w − 4)²，w = 2 时 L = {L.item():.0f}，dL/dw = {w.grad.item():.0f}（手算 2 × (3×2 − 4) × 3 = 12）")

# ---------- Keras 3 ----------
keras.utils.set_random_seed(42)
kmodel = keras.Sequential([
    keras.Input(shape=(64,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
kmodel.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print(f"\n② Keras 3（后端 {keras.backend.backend()}）：参数个数 {kmodel.count_params()}")
t = time.perf_counter()
hist = kmodel.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH, validation_data=(X_val, y_val), verbose=0)
t_keras = time.perf_counter() - t
for epoch in (1, 5, 10, 20, 30):
    print(f"  第 {epoch:>2} 轮：训练损失 {hist.history['loss'][epoch - 1]:.4f}，验证准确率 {hist.history['val_accuracy'][epoch - 1]:.3f}")
acc_keras = kmodel.evaluate(X_test, y_test, verbose=0)[1]
print(f"  测试准确率 {acc_keras:.3f}，训练用时 {t_keras:.1f} 秒")

print("\n③ 对比（同一份数据、同样的网络结构和训练设置）")
print(f"  PyTorch  测试准确率 {acc_torch:.3f}   用时 {t_torch:.1f} 秒")
print(f"  Keras 3  测试准确率 {acc_keras:.3f}   用时 {t_keras:.1f} 秒")
print("  第 6 章 SVM（RBF 核）测试准确率 0.989；逻辑回归 0.981")
