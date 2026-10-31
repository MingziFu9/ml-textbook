"""实验 9-1 交叉验证与学习曲线

① 比较一次划分和 5 折交叉验证的稳定性；比较普通 K 折、分层 K 折每一折的流失率；演示时间序列交叉验证的划分方式。
② 用学习曲线诊断两个模型：第 3 章的二手房多元线性回归（欠拟合）、第 5 章不限深度的决策树（过拟合）。
③ 用验证曲线为决策树选择最大深度，再用学习曲线检查改进效果。
运行：python lab9_1_cv_learning_curve.py（在 labs/ch09 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score,
                                     learning_curve, train_test_split, validation_curve)
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# 客户流失数据（第 4、5 章）
churn = pd.read_csv("../data/telco_churn.csv")
yc = (churn["是否流失"] == "是").astype(int)
Xc = pd.get_dummies(churn.drop(columns=["客户编号", "是否流失"]),
                    columns=["合约类型", "宽带类型", "技术支持", "视频会员", "付款方式"], dtype=int)

# 二手房数据（第 3 章）
house = pd.read_csv("../data/house_clean.csv").dropna(subset=["面积", "建成年份", "总价_万元"])
house = house.sort_values("挂牌日期").reset_index(drop=True)
Xh = house[["面积", "建成年份", "室", "厅", "总楼层"]].join(
    pd.get_dummies(house[["区域", "装修", "朝向"]], drop_first=True, dtype=int)).fillna(0)
yh = house["总价_万元"]
print(f"客户流失 {len(Xc)} 条（流失率 {yc.mean():.1%}）；二手房 {len(Xh)} 套")

# ① 一次划分 vs 交叉验证
tree5 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
one = [tree5.fit(a, c).score(b, d) for a, b, c, d in
       (train_test_split(Xc, yc, test_size=0.3, random_state=s, stratify=yc) for s in range(10))]
print(f"\n① 深度 5 的决策树：10 次不同的随机划分，测试准确率 {min(one):.3f}～{max(one):.3f}（相差 {max(one) - min(one):.3f}）")
cv = cross_val_score(tree5, Xc, yc, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print("  5 折交叉验证每一折：" + "，".join(f"{s:.3f}" for s in cv) + f"；平均 {cv.mean():.3f}，标准差 {cv.std():.3f}")

small = churn.sample(200, random_state=1)
ys = (small["是否流失"] == "是").astype(int).to_numpy()
for name, kf in [("普通 K 折", KFold(5)), ("分层 K 折", StratifiedKFold(5))]:
    rates = [ys[te].mean() for _, te in kf.split(small, ys)]
    print(f"  抽 200 条（流失率 {ys.mean():.1%}），{name}每一折验证集的流失率：" + "，".join(f"{r:.1%}" for r in rates))

months = pd.period_range("2025-01", periods=12, freq="M")
print("  时间序列交叉验证（12 个月，分 4 折）：")
for i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=4, test_size=2).split(months), 1):
    print(f"    第 {i} 折：训练 {months[tr[0]]}～{months[tr[-1]]}，验证 {months[te[0]]}～{months[te[-1]]}")

# ② 学习曲线
sizes = np.linspace(0.1, 1.0, 8)


def lc(model, X, y, scoring, cv):
    n, tr, va = learning_curve(model, X, y, train_sizes=sizes, cv=cv, scoring=scoring, n_jobs=-1)
    return n, tr.mean(1), va.mean(1)


kf = KFold(5, shuffle=True, random_state=42)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
curves = {
    "二手房 · 多元线性回归（第 3 章）": lc(LinearRegression(), Xh, yh, "r2", kf),
    "二手房 · 梯度提升树": lc(HistGradientBoostingRegressor(random_state=42), Xh, yh, "r2", kf),
    "客户流失 · 决策树（不限深度）": lc(DecisionTreeClassifier(random_state=42), Xc, yc, "accuracy", skf),
}
print("\n② 学习曲线（5 折交叉验证）")
for name, (n, tr, va) in curves.items():
    print(f"  {name}")
    for k in [0, 3, 7]:
        print(f"    训练样本 {n[k]:>5}：训练得分 {tr[k]:.3f}，验证得分 {va[k]:.3f}，差距 {tr[k] - va[k]:.3f}")

# ③ 验证曲线：决策树最大深度
depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
tr, va = validation_curve(DecisionTreeClassifier(min_samples_leaf=1, random_state=42), Xc, yc,
                          param_name="max_depth", param_range=depths, cv=skf, n_jobs=-1)
print("\n③ 验证曲线：决策树的最大深度")
for d, a, b in zip(depths, tr.mean(1), va.mean(1)):
    print(f"  深度 {d:>2}：训练 {a:.3f}，验证 {b:.3f}")
best = depths[int(np.argmax(va.mean(1)))]
print(f"  验证得分最高的深度 = {best}")
curves[f"客户流失 · 决策树（深度 {best}）"] = lc(DecisionTreeClassifier(max_depth=best, random_state=42), Xc, yc, "accuracy", skf)
n, a, b = curves[f"客户流失 · 决策树（深度 {best}）"]
print(f"  深度 {best} 的学习曲线：全部训练样本时训练 {a[-1]:.3f}，验证 {b[-1]:.3f}，差距 {a[-1] - b[-1]:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, (name, (n, a, b)) in zip(axes, curves.items()):
    ax.plot(n, a, "o-", label="训练得分"); ax.plot(n, b, "s-", label="验证得分")
    ax.set_title(name, fontsize=10); ax.set_xlabel("训练样本数"); ax.grid(alpha=0.3)
axes[0].set_ylabel("R² / 准确率"); axes[0].legend()
plt.tight_layout(); plt.savefig("lab9_1_learning_curves.png", dpi=110)

plt.figure(figsize=(6, 3.6))
plt.plot(depths, tr.mean(1), "o-", label="训练"); plt.plot(depths, va.mean(1), "s-", label="验证")
plt.axvline(best, color="gray", ls="--", lw=1); plt.xlabel("最大深度 max_depth"); plt.ylabel("准确率")
plt.title("验证曲线：决策树的最大深度"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("lab9_1_validation_curve.png", dpi=120)
