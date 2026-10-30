"""实验 6-3 手写数字识别

scikit-learn 自带的 8×8 手写数字（1797 张）：比较 SVM、逻辑回归、随机森林的准确率和用时，
查看混淆矩阵和被认错的数字图片，再看训练样本少时三种模型的表现。
运行：python lab6_3_digits.py（在 labs/ch06 目录下）
"""
import logging
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

digits = load_digits()
X, y = digits.data, digits.target
print(f"共 {len(y)} 张图片，每张 8×8 = {X.shape[1]} 个像素，灰度 0～16；各数字的张数：{np.bincount(y).tolist()}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集 {len(y_train)} 张，测试集 {len(y_test)} 张")

models = {"SVM（RBF 核）": SVC(),
          "SVM（RBF 核，标准化）": make_pipeline(StandardScaler(), SVC()),
          "逻辑回归（标准化）": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
          "随机森林（300 棵）": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)}
print("\n① 三种模型对比")
print(f"  {'模型':<20}{'训练准确率':>8}{'测试准确率':>8}{'认错张数':>8}{'训练用时（秒）':>10}")
for name, m in models.items():
    t = time.perf_counter()
    m.fit(X_train, y_train)
    sec = time.perf_counter() - t
    print(f"  {name:<20}{m.score(X_train, y_train):>10.3f}{m.score(X_test, y_test):>11.3f}"
          f"{int((m.predict(X_test) != y_test).sum()):>9}{sec:>13.2f}")
svm = models["SVM（RBF 核）"]
print(f"  SVM 的 gamma 默认取 'scale' = 1 / (64 × 像素方差) = {1 / (64 * X_train.var()):.5f}；"
      f"支持向量 {len(svm.support_)} 个，占训练集 {len(svm.support_) / len(y_train):.0%}；各数字的支持向量数 {svm.n_support_.tolist()}")

pred = svm.predict(X_test)
cm = confusion_matrix(y_test, pred)
print("\n② SVM 的混淆矩阵（行：真实数字，列：预测数字）")
print("     " + "".join(f"{d:>4}" for d in range(10)))
for d in range(10):
    print(f"  {d}: " + "".join(f"{v:>4}" for v in cm[d]))
wrong = np.where(pred != y_test)[0]
print(f"  认错的 {len(wrong)} 张：" + "，".join(f"{y_test[i]}→{pred[i]}" for i in wrong))

fig, axes = plt.subplots(1, len(wrong), figsize=(1.6 * len(wrong), 2))
for ax, i in zip(np.atleast_1d(axes), wrong):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"真 {y_test[i]} → 判 {pred[i]}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("lab6_3_wrong.png", dpi=120)

print("\n③ 训练样本很少时（从训练集中每个数字只取 n 张，测试集不变）")
print("  每个数字    SVM     逻辑回归   随机森林")
rng = np.random.default_rng(0)
for n in [5, 10, 20, 50]:
    idx = np.concatenate([rng.choice(np.where(y_train == d)[0], n, replace=False) for d in range(10)])
    accs = [m.fit(X_train[idx], y_train[idx]).score(X_test, y_test)
            for m in [SVC(), make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                      RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)]]
    print(f"  {n:>6} 张" + "".join(f"{a:>10.3f}" for a in accs))
