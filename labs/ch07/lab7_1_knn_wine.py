"""实验 7-1 K 近邻葡萄酒分类与 K 值选择

葡萄酒数据（scikit-learn 自带，178 条，13 项化学成分，3 个品种）：比较标准化前后 KNN 的效果，
手算一个样本的近邻，画出 K 值与准确率的曲线，比较三种距离，再观察加入无关特征后的“维度灾难”。
运行：python lab7_1_knn_wine.py（在 labs/ch07 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
print(f"共 {len(y)} 条，{X.shape[1]} 个特征，3 个品种的条数：{np.bincount(y).tolist()}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集 {len(y_train)} 条，测试集 {len(y_test)} 条")
rng = X_train.agg(["min", "max", "std"]).T.sort_values("std")
print("\n① 特征的取值范围相差很大（训练集，按标准差排序，取首尾各 3 个）：")
print(pd.concat([rng.head(3), rng.tail(3)]).round(2).to_string())

print("\n② K = 5，标准化前后对比")
raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
std = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_train, y_train)
print(f"  不标准化：测试准确率 {raw.score(X_test, y_test):.3f}")
print(f"  标准化后：测试准确率 {std.score(X_test, y_test):.3f}")

# 手算：测试集第 1 条的 5 个近邻
scaler, knn = std[0], std[-1]
Z_train, z0 = scaler.transform(X_train), scaler.transform(X_test.iloc[[0]])[0]
d = np.sqrt(((Z_train - z0) ** 2).sum(axis=1))
idx = np.argsort(d)[:5]
print(f"\n③ 测试集第 1 条（真实品种 {y_test[0]}）在标准化空间中的 5 个最近邻：")
for r, i in enumerate(idx, 1):
    print(f"  第 {r} 近：训练样本 {X_train.index[i]:>3}，距离 {d[i]:.2f}，品种 {y_train[i]}")
votes = np.bincount(y_train[idx], minlength=3)
print(f"  投票：品种 0/1/2 各 {votes.tolist()} 票 → 判为品种 {votes.argmax()}；predict 给出 {std.predict(X_test.iloc[[0]])[0]}")
d_raw = np.sqrt(((X_train.to_numpy() - X_test.iloc[0].to_numpy()) ** 2).sum(axis=1))
idx_raw = np.argsort(d_raw)[:5]
share = ((X_train.iloc[idx_raw]["proline"] - X_test.iloc[0]["proline"]) ** 2 / d_raw[idx_raw] ** 2).mean()
print(f"  不标准化时，5 个最近邻的品种是 {y_train[idx_raw].tolist()}；距离的平方中 proline 一项平均占 {share:.0%}，"
      f"相当于只按 proline 找邻居")
print(f"  不标准化时整个测试集：{(raw.predict(X_test) != y_test).sum()} 条判错；标准化后 {(std.predict(X_test) != y_test).sum()} 条")

print("\n④ K 值与准确率（标准化；交叉验证只用训练集）")
ks = list(range(1, 31))
tr, cv = [], []
for k in ks:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv.append(cross_val_score(m, X_train, y_train, cv=5).mean())
    tr.append(m.fit(X_train, y_train).score(X_train, y_train))
for k in [1, 3, 5, 7, 9, 15, 21, 30]:
    print(f"  K = {k:>2}：训练准确率 {tr[k - 1]:.3f}，5 折交叉验证准确率 {cv[k - 1]:.3f}")
best_k = ks[int(np.argmax(cv))]
m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X_train, y_train)
print(f"  交叉验证最好的 K = {best_k}（{max(cv):.3f}），测试准确率 {m.score(X_test, y_test):.3f}")
plt.figure(figsize=(7, 3.6))
plt.plot(ks, tr, "o-", ms=3, label="训练准确率")
plt.plot(ks, cv, "s-", ms=3, label="5 折交叉验证准确率")
plt.axvline(best_k, color="gray", ls="--", lw=1)
plt.xlabel("K（近邻个数）"), plt.ylabel("准确率"), plt.legend(), plt.title("K 值与准确率（葡萄酒数据，标准化）")
plt.tight_layout()
plt.savefig("lab7_1_k.png", dpi=120)

print(f"\n⑤ 三种距离（标准化，K = {best_k}）")
for metric in ["euclidean", "manhattan", "cosine"]:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k, metric=metric))
    cvs = cross_val_score(m, X_train, y_train, cv=5).mean()
    print(f"  {metric:<10} 交叉验证 {cvs:.3f}，测试准确率 {m.fit(X_train, y_train).score(X_test, y_test):.3f}")

print(f"\n⑥ 维度灾难：加入纯随机的无关特征（标准化，K = {best_k}）")
noise_rng = np.random.default_rng(0)
for extra in [0, 10, 50, 200, 1000]:
    Xn = np.hstack([X.to_numpy(), noise_rng.normal(size=(len(X), extra))])
    a, b, ya, yb = train_test_split(Xn, y, test_size=0.3, random_state=42, stratify=y)
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(a, ya)
    Za = m[0].transform(a)
    dist = np.sqrt(((Za[:, None, :] - Za[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(Za), 1)]
    print(f"  特征 {13 + extra:>4} 个：测试准确率 {m.score(b, yb):.3f}；样本间距离 最小 {dist.min():.1f}，"
          f"平均 {dist.mean():.1f}，最小/平均 = {dist.min() / dist.mean():.2f}")
