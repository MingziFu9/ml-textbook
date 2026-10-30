"""实验 6-1 线性 SVM 与参数 C

第一部分：二维示例数据上训练线性 SVM，画出决策边界、间隔和支持向量，观察惩罚参数 C 从小到大的变化，
再加入一个“异常点”，比较 C 大、C 小时边界受影响的程度。
第二部分：贷款审批数据（与第 5 章同一份数据、同样的划分）上比较不标准化和标准化的 SVM。
运行：python lab6_1_linear_svm.py（在 labs/ch06 目录下）
"""
import logging
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def draw(ax, model, X, y, title):
    """画样本点、决策边界（实线）、间隔边界（虚线），圈出支持向量"""
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=22, edgecolors="k", linewidths=0.4)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
    f = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, f, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=1)
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none", edgecolors="green", linewidths=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])


# ---------- 第一部分：二维数据上的线性 SVM ----------
X, y = make_blobs(n_samples=60, centers=2, cluster_std=1.3, random_state=4)
print(f"二维示例数据：{len(y)} 个点，两类各 {np.bincount(y)[0]}、{np.bincount(y)[1]} 个")
print("\n① 惩罚参数 C 对间隔和支持向量的影响")
print("     C   支持向量数   间隔宽度 2/‖w‖   训练集分错的点")
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, C in zip(axes, [0.01, 0.1, 1, 100]):
    m = SVC(kernel="linear", C=C).fit(X, y)
    width = 2 / np.linalg.norm(m.coef_[0])
    wrong = int((m.predict(X) != y).sum())
    print(f"{C:>6}   {len(m.support_):>6}       {width:>8.2f}        {wrong:>5}")
    draw(ax, m, X, y, f"C = {C}：支持向量 {len(m.support_)} 个，间隔 {width:.2f}")
plt.tight_layout()
plt.savefig("lab6_1_C.png", dpi=120)

m = SVC(kernel="linear", C=1).fit(X, y)
w, b = m.coef_[0], m.intercept_[0]
print(f"\n② C = 1 时：w = ({w[0]:.3f}, {w[1]:.3f})，b = {b:.3f}，‖w‖ = {np.linalg.norm(w):.3f}，"
      f"间隔宽度 = 2 / {np.linalg.norm(w):.3f} = {2 / np.linalg.norm(w):.2f}")
f_sv = m.decision_function(m.support_vectors_)
on_margin = np.abs(np.abs(f_sv) - 1) < 1e-3
print(f"   {len(f_sv)} 个支持向量中，{on_margin.sum()} 个正好落在间隔边界上（f(x) = ±1），"
      f"其余 {(~on_margin).sum()} 个在间隔内或被分错（|f(x)| < 1）")
x0 = m.support_vectors_[np.argmax(on_margin)]
print(f"   例：支持向量 ({x0[0]:.2f}, {x0[1]:.2f})，f(x) = {w[0]:.3f}×{x0[0]:.2f} + {w[1]:.3f}×{x0[1]:.2f} "
      f"+ ({b:.3f}) = {w @ x0 + b:.3f}")
far = X[np.argmax(np.abs(m.decision_function(X)))]
print(f"   离边界最远的点 ({far[0]:.2f}, {far[1]:.2f})，f(x) = {w @ far + b:.3f}，不是支持向量，删掉它边界不变")

# 加入一个异常点：在 1 类的区域里放一个 0 类的点
Xo, yo = make_blobs(n_samples=60, centers=2, cluster_std=1.3, random_state=6)
c0, c1 = Xo[yo == 0].mean(axis=0), Xo[yo == 1].mean(axis=0)
outlier = c0 + 0.55 * (c1 - c0)          # 在两类中心的连线上、略偏向 1 类的位置
X2, y2 = np.vstack([Xo, outlier]), np.append(yo, 0)
print(f"\n③ 另一份可以完全分开的数据，再加入 1 个异常点 ({outlier[0]:.2f}, {outlier[1]:.2f})（标为 0 类，却靠近 1 类）")
print("            C    支持向量数   间隔宽度   训练集分错   异常点被分对了吗")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (data, C, name) in zip(axes, [((Xo, yo), 1, "无异常点"), ((X2, y2), 1000, "有异常点"),
                                       ((X2, y2), 0.1, "有异常点")]):
    Xa, ya = data
    mm = SVC(kernel="linear", C=C).fit(Xa, ya)
    width = 2 / np.linalg.norm(mm.coef_[0])
    wrong = int((mm.predict(Xa) != ya).sum())
    ok = "—" if name == "无异常点" else ("是" if mm.predict([outlier])[0] == 0 else "否")
    print(f"  {name}  {C:>6}   {len(mm.support_):>6}     {width:>7.2f}     {wrong:>5}        {ok}")
    draw(ax, mm, Xa, ya, f"{name}，C = {C}：间隔 {width:.2f}")
    if name != "无异常点":
        ax.scatter(*outlier, marker="*", s=260, c="gold", edgecolors="k", zorder=5)
plt.tight_layout()
plt.savefig("lab6_1_outlier.png", dpi=120)

# ---------- 第二部分：贷款审批数据 ----------
df = pd.read_csv("../data/loan_apply.csv")
df["有房产"] = (df["有房产"] == "是").astype(int)
X = pd.get_dummies(df.drop(columns="是否批准"), columns=["信用等级"], dtype=int)
y = (df["是否批准"] == "是").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\n④ 贷款审批数据：训练集 {len(X_train)} 条，测试集 {len(X_test)} 条，特征 {X.shape[1]} 个")
print(X_train[["月收入", "负债率", "近两年逾期次数", "申请金额_万元"]].describe().loc[["mean", "std", "min", "max"]].round(2))
print("\n  模型                         训练准确率  测试准确率  支持向量数  用时（秒）")
for name, model in [("线性核 SVM，标准化", make_pipeline(StandardScaler(), SVC(kernel="linear"))),
                    ("RBF 核 SVM，不标准化", SVC()),
                    ("RBF 核 SVM，标准化", make_pipeline(StandardScaler(), SVC()))]:
    t = time.perf_counter()
    model.fit(X_train, y_train)
    sec = time.perf_counter() - t
    svc = model[-1] if hasattr(model, "steps") else model
    print(f"  {name:<22}{model.score(X_train, y_train):>8.3f}{model.score(X_test, y_test):>10.3f}"
          f"{len(svc.support_):>10}{sec:>11.2f}")
