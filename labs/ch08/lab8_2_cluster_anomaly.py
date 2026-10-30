"""实验 8-2 三种聚类算法对比与异常检测

第一部分：在三种形状的二维数据（球形、月牙形、密度不同的球形）上比较 K-Means、层次聚类和 DBSCAN，
用调整兰德指数（ARI，越接近 1 越和真实分组一致）评价，并画出层次聚类的树状图。
第二部分：在信用卡交易数据上，不看标签，用孤立森林和 3σ 法则找出可疑交易，再用标签检验效果。
运行：python lab8_2_cluster_anomaly.py（在 labs/ch08 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 第一部分：三种聚类算法 ----------
blobs = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=1)
moons = make_moons(n_samples=300, noise=0.06, random_state=42)
Xv, yv = make_blobs(n_samples=[200, 60, 40], centers=[[0, 0], [5, 5], [6, -2]], cluster_std=[1.6, 0.4, 0.4], random_state=3)
data = {"球形": blobs, "月牙形": moons, "密度不同": (Xv, yv)}
params = {"球形": 0.3, "月牙形": 0.3, "密度不同": 0.3}          # DBSCAN 的 eps（数据已标准化）

print("① 三种算法的调整兰德指数 ARI（1 = 与真实分组完全一致，0 ≈ 随机分组）")
print(f"  {'数据':<8}{'K-Means':>9}{'层次聚类':>9}{'DBSCAN':>9}   DBSCAN 找到的簇数 / 噪声点数")
fig, axes = plt.subplots(3, 3, figsize=(11, 10))
for r, (name, (X, y)) in enumerate(data.items()):
    X = StandardScaler().fit_transform(X)
    k = len(np.unique(y))
    models = {"K-Means": KMeans(n_clusters=k, n_init=10, random_state=42),
              "层次聚类（单链接）" if name == "月牙形" else "层次聚类（Ward）":
                  AgglomerativeClustering(n_clusters=k, linkage="single" if name == "月牙形" else "ward"),
              "DBSCAN": DBSCAN(eps=params[name], min_samples=5)}
    aris = []
    for c, (mname, m) in enumerate(models.items()):
        lab = m.fit_predict(X)
        aris.append(adjusted_rand_score(y, lab))
        ax = axes[r, c]
        ax.scatter(X[lab >= 0, 0], X[lab >= 0, 1], c=lab[lab >= 0], cmap="tab10", s=8)
        ax.scatter(X[lab < 0, 0], X[lab < 0, 1], c="k", marker="x", s=14)
        ax.set_title(f"{name} · {mname}：ARI {aris[-1]:.2f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    lab = models["DBSCAN"].labels_
    print(f"  {name:<8}" + "".join(f"{a:>10.3f}" for a in aris) +
          f"     {len(set(lab)) - (1 if -1 in lab else 0)} 簇 / {(lab == -1).sum()} 个")
plt.tight_layout()
plt.savefig("lab8_2_compare.png", dpi=110)

# eps 的影响
X = StandardScaler().fit_transform(moons[0])
print("\n② DBSCAN 在月牙形数据上，eps 的影响（min_samples = 5）")
for eps in [0.1, 0.2, 0.3, 0.5, 1.0]:
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    print(f"  eps = {eps:<4} 簇数 {len(set(lab)) - (1 if -1 in lab else 0):>2}，噪声点 {(lab == -1).sum():>3}，"
          f"ARI {adjusted_rand_score(moons[1], lab):.3f}")

# 层次聚类树状图：取球形数据的前 30 个点，便于看清
Xs = StandardScaler().fit_transform(blobs[0])[:30]
plt.figure(figsize=(9, 3.2))
dendrogram(linkage(Xs, method="ward"), color_threshold=3, labels=[str(i) for i in range(30)])
plt.axhline(3, color="gray", ls="--", lw=1)
plt.title("层次聚类树状图（球形数据前 30 个点，Ward 方法；在虚线高度切一刀得到 3 个簇）")
plt.tight_layout()
plt.savefig("lab8_2_dendrogram.png", dpi=120)

# ---------- 第二部分：异常检测 ----------
df = pd.read_csv("../data/card_fraud.csv")
feats = ["交易金额", "交易小时", "是否境外", "是否新设备", "距上次交易_小时", "近24小时交易次数", "距常用地_公里"]
X = df[feats].copy()
X["交易金额"] = np.log1p(X["交易金额"])
X["距常用地_公里"] = np.log1p(X["距常用地_公里"])
X["深夜"] = X.pop("交易小时").between(0, 5).astype(int)
y = df["是否欺诈"].to_numpy()
print(f"\n③ 信用卡交易 {len(df)} 笔（欺诈 {y.sum()} 笔，{y.mean():.2%}）；孤立森林不使用标签")

iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42).fit(X)
score = -iso.score_samples(X)                         # 越大越异常
flag = iso.predict(X) == -1
print(f"  孤立森林（contamination = 0.01）标出 {flag.sum()} 笔可疑交易，其中真欺诈 {y[flag].sum()} 笔，"
      f"命中率 {y[flag].mean():.1%}（随机抽取约 {y.mean():.1%}），找到了全部欺诈的 {y[flag].sum() / y.sum():.1%}")
order = np.argsort(-score)
for k in [50, 200, 1000]:
    print(f"  异常分数最高的 {k:>4} 笔中，欺诈 {y[order[:k]].sum():>3} 笔（{y[order[:k]].mean():.1%}）")

z = (np.log1p(df["交易金额"]) - np.log1p(df["交易金额"]).mean()) / np.log1p(df["交易金额"]).std()
f3 = (z.abs() > 3).to_numpy()
print(f"  对比：只看交易金额的 3σ 法则标出 {f3.sum()} 笔，其中欺诈 {y[f3].sum()} 笔（{y[f3].mean():.1%}）")
top = df.loc[order[:5], ["交易编号", "交易金额", "交易小时", "商户类别", "是否境外", "是否新设备", "距常用地_公里", "是否欺诈"]]
print("  异常分数最高的 5 笔：")
print(top.to_string(index=False))
