"""实验 3-3 过拟合与正则化

奶茶店的日销量与当天气温：只有 20 天的训练数据。多项式阶数从 1 加到 15 观察欠拟合和过拟合，
再在 15 阶多项式上用岭回归和 Lasso 治理，比较系数的变化。
运行：python lab3_3_regularization.py（在 labs/ch03 目录下）
"""
import json
import logging
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# 模拟数据：销量随气温先升后降（30 ℃ 左右最好卖，太热了人们不愿出门），加上随机波动
rng = np.random.default_rng(3)


def sales(t):
    return 100 + 12 * t - 0.2 * t ** 2


t_train = np.sort(rng.uniform(5, 38, 20))
t_test = np.sort(rng.uniform(t_train.min(), t_train.max(), 200))   # 测试日的气温在训练数据的范围内
y_train = sales(t_train) + rng.normal(0, 20, t_train.size)
y_test = sales(t_test) + rng.normal(0, 20, t_test.size)
t_train, t_test = t_train.round(2), t_test.round(2)          # 气温保留 2 位小数、销量保留 1 位，与交互动画用的数据完全一致
y_train, y_test = y_train.round(1), y_test.round(1)
X_train, X_test = t_train.reshape(-1, 1), t_test.reshape(-1, 1)
grid = np.linspace(t_train.min(), t_train.max(), 300).reshape(-1, 1)
print(f"训练集 {len(t_train)} 天，测试集 {len(t_test)} 天；真实规律：销量 = 100 + 12 × 气温 − 0.2 × 气温²，随机波动标准差 20")


def poly_model(degree, reg=None):
    """先把气温缩放到 [-1, 1]，再生成 1～degree 次方，标准化后接回归模型"""
    return make_pipeline(MinMaxScaler((-1, 1)), PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), reg or LinearRegression())


print("\n① 多项式阶数与过拟合")
print("  阶数   训练 MSE   测试 MSE")
rows, curves = [], {}
for d in range(1, 16):
    m = poly_model(d).fit(X_train, y_train)
    tr, te = mean_squared_error(y_train, m.predict(X_train)), mean_squared_error(y_test, m.predict(X_test))
    rows.append((d, tr, te))
    curves[d] = m.predict(grid)
    print(f"  {d:>3}{tr:>11.1f}{te:>12.1f}")
best = min(rows, key=lambda r: r[2])
print(f"  测试 MSE 最小的是 {best[0]} 阶（{best[2]:.1f}）；随机波动本身的方差是 400")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(t_train, y_train, color="black", zorder=3, label="训练数据（20 天）")
for d, c in [(1, "#1e88e5"), (2, "#43a047"), (15, "#e53935")]:
    axes[0].plot(grid, np.clip(curves[d], 0, 500), color=c, label=f"{d} 阶")
axes[0].set(xlabel="气温（℃）", ylabel="日销量（杯）", title="多项式阶数：欠拟合、刚好、过拟合", ylim=(0, 450))
axes[0].legend()
axes[1].plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="训练 MSE")
axes[1].plot([r[0] for r in rows], [r[2] for r in rows], "s-", label="测试 MSE")
axes[1].set(xlabel="多项式阶数", ylabel="均方误差（对数刻度）", yscale="log", title="阶数越高，训练误差越小，测试误差先降后升")
axes[1].legend()
plt.tight_layout()
plt.savefig("lab3_3_degree.png", dpi=150)

print("\n② 15 阶多项式 + 正则化")
print("  模型                 alpha    训练 MSE   测试 MSE   系数绝对值之和   为 0 的系数")
models = [("普通线性回归", None, LinearRegression())]
models += [("岭回归", a, Ridge(alpha=a)) for a in [0.01, 0.1, 1, 10]]
models += [("Lasso", a, Lasso(alpha=a, max_iter=100000)) for a in [0.1, 1, 5]]
for name, a, reg in models:
    m = poly_model(15, reg).fit(X_train, y_train)
    coef = m[-1].coef_
    tr, te = mean_squared_error(y_train, m.predict(X_train)), mean_squared_error(y_test, m.predict(X_test))
    print(f"  {name:<12}{'—' if a is None else a:>8}{tr:>11.1f}{te:>12.1f}{np.abs(coef).sum():>14.1f}{(np.abs(coef) < 1e-8).sum():>10}")
    if name == "Lasso" and a == 1:
        lasso_coef = coef

print("  Lasso（alpha=1）各次方的系数：", {f"x^{i + 1}": round(float(c), 2) for i, c in enumerate(lasso_coef) if abs(c) > 1e-8})

# 导出数据给交互动画“多项式阶数与过拟合”使用
json.dump({"train": [[float(t), float(y)] for t, y in zip(t_train, y_train)],
           "test": [[float(t), float(y)] for t, y in zip(t_test, y_test)]},
          open("lab3_3_data.json", "w"), ensure_ascii=False)
