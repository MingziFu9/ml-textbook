"""实验 8-3 PCA 与 t-SNE 降维可视化

手写数字（1797 张 8×8 图片，64 个特征）：用 PCA 计算方差解释率，降到二维画图；用 t-SNE 降到二维画图对比；
再比较降维前后 SVM 分类器的速度和准确率；最后看看主成分“长什么样”，以及只用少数主成分还原的图片。
运行：python lab8_3_pca_tsne.py（在 labs/ch08 目录下），t-SNE 一步约需 10 秒
"""
import logging
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

X, y = load_digits(return_X_y=True)
print(f"手写数字 {X.shape[0]} 张，每张 {X.shape[1]} 个像素特征")

# ① PCA 方差解释率（像素单位相同，这里不做标准化）
pca = PCA().fit(X)
ratio = pca.explained_variance_ratio_
cum = np.cumsum(ratio)
print("\n① 前 10 个主成分的方差解释率：" + "，".join(f"{r:.3f}" for r in ratio[:10]))
for k in [2, 10, 20, 30]:
    print(f"  前 {k:>2} 个主成分累计解释方差 {cum[k - 1]:.1%}")
for t in [0.8, 0.9, 0.95]:
    print(f"  累计达到 {t:.0%} 需要 {np.argmax(cum >= t) + 1} 个主成分")
print(f"  有 {(pca.explained_variance_ < 1e-10).sum()} 个方向的方差几乎为 0（图片边缘总是空白的像素）")

# ② 降到二维：PCA 与 t-SNE
Z_pca = PCA(n_components=2).fit_transform(X)
t = time.perf_counter()
Z_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
t_tsne = time.perf_counter() - t
print(f"\n② PCA 降到二维（保留 {cum[1]:.1%} 的方差），t-SNE 降到二维用时 {t_tsne:.1f} 秒")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].plot(range(1, 65), cum, "o-", ms=3); ax[0].axhline(0.9, color="gray", ls="--", lw=1)
ax[0].set_xlabel("主成分个数"), ax[0].set_ylabel("累计方差解释率"), ax[0].set_title("PCA 累计方差解释率")
for a, Z, title in [(ax[1], Z_pca, "PCA 二维"), (ax[2], Z_tsne, "t-SNE 二维")]:
    sc = a.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5)
    for d in range(10):
        cx, cy = np.median(Z[y == d], axis=0)
        a.text(cx, cy, str(d), fontsize=14, weight="bold", ha="center", va="center",
               bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    a.set_title(title), a.set_xticks([]), a.set_yticks([])
plt.tight_layout()
plt.savefig("lab8_3_pca_tsne.png", dpi=110)


def knn_purity(Z, k=10):
    """每个点的 10 个最近邻中，与它数字相同的比例：衡量二维图上同类聚得紧不紧"""
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    return (y[idx] == y[:, None]).mean()


print(f"  二维图上每个点的 10 个最近邻里同一数字的比例：PCA {knn_purity(Z_pca):.3f}，t-SNE {knn_purity(Z_tsne):.3f}")

# ③ 降维后再分类
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("\n③ 降维后再用 SVM 分类（PCA 只在训练集上拟合）")
print("  特征数    测试准确率   训练 + 预测用时（秒）")
for k in [2, 5, 10, 20, 30, 64]:
    model = SVC() if k == 64 else make_pipeline(PCA(n_components=k), SVC())
    t = time.perf_counter()
    acc = model.fit(X_train, y_train).score(X_test, y_test)
    print(f"  {k:>4} 个   {acc:>8.3f}   {time.perf_counter() - t:>12.3f}")

# ④ 主成分长什么样，以及用少数主成分还原图片
fig, axes = plt.subplots(2, 6, figsize=(10, 3.8))
axes[0, 0].imshow(pca.mean_.reshape(8, 8), cmap="gray_r"); axes[0, 0].set_title("平均图片", fontsize=9)
for i in range(5):
    axes[0, i + 1].imshow(pca.components_[i].reshape(8, 8), cmap="RdBu_r")
    axes[0, i + 1].set_title(f"第 {i + 1} 主成分（{ratio[i]:.1%}）", fontsize=9)
img = X[0]
axes[1, 0].imshow(img.reshape(8, 8), cmap="gray_r"); axes[1, 0].set_title("原图（数字 0）", fontsize=9)
errs = []
for j, k in enumerate([2, 5, 10, 20, 40]):
    p = PCA(n_components=k).fit(X)
    rec = p.inverse_transform(p.transform(img[None]))[0]
    errs.append((k, np.abs(rec - img).mean()))
    axes[1, j + 1].imshow(rec.reshape(8, 8), cmap="gray_r"); axes[1, j + 1].set_title(f"用 {k} 个主成分还原", fontsize=9)
for a in axes.ravel():
    a.set_xticks([]), a.set_yticks([])
plt.tight_layout()
plt.savefig("lab8_3_components.png", dpi=120)
print("\n④ 用前 k 个主成分还原第 1 张图片（数字 0）的平均绝对误差（像素灰度 0～16）：" +
      "，".join(f"{k} 个 {e:.2f}" for k, e in errs))
