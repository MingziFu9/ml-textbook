"""实验 6-2 核函数与非线性分类

月牙形、同心圆两份数据上比较线性核、多项式核、RBF 核；手工“抬高一维”体会核技巧的思路；
在月牙数据上用 5 折交叉验证观察 C 与 gamma 的组合；最后用支持向量回归（SVR）拟合第 3 章奶茶店的销量数据。
运行：python lab6_2_kernels.py（在 labs/ch06 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def draw(ax, model, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 250),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 250))
    z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, z > 0, alpha=0.18, cmap="coolwarm")
    ax.contour(xx, yy, z, levels=[0], colors="k", linewidths=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10, edgecolors="k", linewidths=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])


datasets = {"月牙形": make_moons(n_samples=300, noise=0.2, random_state=42),
            "同心圆": make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)}
kernels = {"线性核": SVC(kernel="linear"),
           "多项式核（3 次）": SVC(kernel="poly", degree=3),
           "RBF 核": SVC(kernel="rbf")}

print("① 三种核函数的测试准确率（每份数据 300 个点，按 7 : 3 划分）")
print(f"  {'数据':<6}" + "".join(f"{k:>14}" for k in kernels))
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for r, (dname, (X, y)) in enumerate(datasets.items()):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    accs = []
    for c, (kname, svc) in enumerate(kernels.items()):
        m = make_pipeline(StandardScaler(), svc).fit(X_train, y_train)
        accs.append(m.score(X_test, y_test))
        draw(axes[r, c], m, X, y, f"{dname} · {kname}：测试准确率 {accs[-1]:.3f}")
    print(f"  {dname:<6}" + "".join(f"{a:>17.3f}" for a in accs))
plt.tight_layout()
plt.savefig("lab6_2_kernels.png", dpi=120)

# 核技巧的思路：给同心圆数据加一个新特征 r² = x1² + x2²，再用线性核
X, y = datasets["同心圆"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
lift = lambda A: np.c_[A, (A ** 2).sum(axis=1)]
m = SVC(kernel="linear").fit(lift(X_train), y_train)
r2 = (X ** 2).sum(axis=1)
print(f"\n② 同心圆数据加一个特征 r² = x1² + x2²（到圆心距离的平方），再用线性核：测试准确率 {m.score(lift(X_test), y_test):.3f}")
print(f"   外圈（0 类）r² 平均 {r2[y == 0].mean():.2f}，内圈（1 类）r² 平均 {r2[y == 1].mean():.2f}；"
      f"三维空间中用一个水平面 r² ≈ {-m.intercept_[0] / m.coef_[0][2]:.2f} 就能切开（w = {m.coef_[0].round(2)}）")

# C 与 gamma 的组合
X, y = datasets["月牙形"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
Cs, gammas = [0.1, 1, 10, 100], [0.1, 1, 10, 100]
print("\n③ 月牙数据：RBF 核不同 C 和 gamma 的训练准确率 / 5 折交叉验证准确率（只用训练集）")
print("  C \\ gamma" + "".join(f"{g:>16}" for g in gammas))
scores = {}
for C in Cs:
    row = []
    for g in gammas:
        m = make_pipeline(StandardScaler(), SVC(C=C, gamma=g))
        cv = cross_val_score(m, X_train, y_train, cv=5).mean()
        tr = m.fit(X_train, y_train).score(X_train, y_train)
        row.append(f"{tr:.3f} / {cv:.3f}")
        scores[(C, g)] = round(cv, 3)
    print(f"  {C:>8}" + "".join(f"{s:>16}" for s in row))
top = max(scores.values())
print(f"  交叉验证准确率最高为 {top:.3f}，并列的组合在测试集上：")
for (C, g), cv in scores.items():
    if cv == top:
        m = make_pipeline(StandardScaler(), SVC(C=C, gamma=g)).fit(X_train, y_train)
        print(f"    C = {C:<4} gamma = {g:<4} 测试准确率 {m.score(X_test, y_test):.3f}，支持向量 {len(m[-1].support_)} 个")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, g in zip(axes, gammas):
    m = make_pipeline(StandardScaler(), SVC(C=1, gamma=g)).fit(X_train, y_train)
    draw(ax, m, X_train, y_train,
         f"gamma = {g}：训练 {m.score(X_train, y_train):.3f}，测试 {m.score(X_test, y_test):.3f}")
plt.tight_layout()
plt.savefig("lab6_2_gamma.png", dpi=120)

# 支持向量回归：第 3 章实验 3-3 的奶茶店数据（同样的随机种子和生成方式）
rng = np.random.default_rng(3)
sales = lambda t: 100 + 12 * t - 0.2 * t ** 2
t_train = np.sort(rng.uniform(5, 38, 20))
t_test = np.sort(rng.uniform(t_train.min(), t_train.max(), 200))
y_train = (sales(t_train) + rng.normal(0, 20, t_train.size)).round(1)
y_test = (sales(t_test) + rng.normal(0, 20, t_test.size)).round(1)
t_train, t_test = t_train.round(2).reshape(-1, 1), t_test.round(2).reshape(-1, 1)
print("\n④ 支持向量回归（RBF 核，特征标准化）拟合奶茶店 20 天的销量，测试集 200 天")
print("     epsilon     C   支持向量数   训练 MSE   测试 MSE")
for eps, C in [(5, 1000), (20, 1000), (40, 1000), (20, 100)]:
    m = make_pipeline(StandardScaler(), SVR(C=C, epsilon=eps)).fit(t_train, y_train)
    print(f"  {eps:>8} {C:>6}   {len(m[-1].support_):>6}   {mean_squared_error(y_train, m.predict(t_train)):>9.1f}"
          f"  {mean_squared_error(y_test, m.predict(t_test)):>9.1f}")
print("  对照（实验 3-3）：2 阶多项式回归测试 MSE 529.4，15 阶 + 岭回归 512.5")
