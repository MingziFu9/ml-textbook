"""实验 3-2 二手房价格预测

在实验 2-2 清洗好的数据上训练多元线性回归，计算 MAE、RMSE、R²，解读系数，画残差图，分析误差最大的 10 套房，
再根据残差图的启发做两处改进。
运行：python lab3_2_house_price.py（在 labs/ch03 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("../data/house_clean.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42)
train, test = train.copy(), test.copy()

# 缺失值用训练集的统计量填充（同实验 2-2）
year_median = train.groupby("区域")["建成年份"].median()
subway_median = train["距地铁站_米"].median()
for part in (train, test):
    part["建成年份"] = part["建成年份"].fillna(part["区域"].map(year_median))
    part["距地铁站_米"] = part["距地铁站_米"].fillna(subway_median)


def features(part):
    X = part[["面积", "建成年份", "室", "厅", "总楼层"]].copy()
    X["距地铁站_千米"] = part["距地铁站_米"] / 1000
    X["楼层位置"] = part["楼层位置"].map({"低": 0, "中": 1, "高": 2})
    # drop_first=True：每个类别特征去掉第一个取值作为“基准”，系数表示与基准相比的差别
    return X.join(pd.get_dummies(part[["区域", "装修", "朝向"]], drop_first=True, dtype=int))


X_train, X_test = features(train), features(test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
y_train, y_test = train["总价_万元"], test["总价_万元"]


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:<22}MAE {mae:6.1f}   RMSE {rmse:6.1f}   R² {r2:6.3f}")
    return mae, rmse, r2


print(f"训练集 {len(X_train)} 套，测试集 {len(X_test)} 套，特征 {X_train.shape[1]} 个；测试集总价均值 {y_test.mean():.1f} 万元")
print("\n① 基线：不管什么房子都猜训练集的平均总价")
report("基线（测试集）", y_test, np.full(len(y_test), y_train.mean()))

print("\n② 多元线性回归")
model = LinearRegression().fit(X_train, y_train)
report("训练集", y_train, model.predict(X_train))
pred = model.predict(X_test)
report("测试集", y_test, pred)

coef = pd.Series(model.coef_, index=X_train.columns)
print(f"\n  截距 {model.intercept_:.1f}；部分系数（区域以“{sorted(df['区域'].unique())[0]}”为基准，装修以“未知”为基准）：")
for c in ["面积", "建成年份", "距地铁站_千米", "楼层位置", "室", "区域_天河", "区域_越秀", "区域_增城", "装修_精装", "装修_毛坯"]:
    print(f"    {c:<10}{coef[c]:>9.2f}")

# 多重共线性：“室”和“面积”高度相关，单独用和一起用时系数差别很大
alone = LinearRegression().fit(X_train[["室"]], y_train).coef_[0]
print(f"\n  “室”与“面积”的相关系数 {train[['室', '面积']].corr().iloc[0, 1]:.3f}；"
      f"只用“室”时系数 {alone:.2f}，和面积等一起用时 {coef['室']:.2f}")

# 残差分析
resid = y_test - pred
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(pred, resid, s=6, alpha=0.5)
axes[0].axhline(0, color="red", lw=1)
axes[0].set(xlabel="预测总价（万元）", ylabel="残差 = 真实值 − 预测值（万元）", title="残差图：线性回归")
print(f"\n  残差：预测价 < 200 万的房子平均绝对残差 {resid[pred < 200].abs().mean():.1f} 万元；> 500 万的 {resid[pred > 500].abs().mean():.1f} 万元")
print(f"  预测值为负数的房子：{(pred < 0).sum()} 套")

worst = test.assign(预测=pred.round(1), 误差=resid.round(1)).reindex(resid.abs().sort_values(ascending=False).index[:10])
print("\n  误差最大的 10 套房：")
print(worst[["区域", "小区", "面积", "建成年份", "装修", "总价_万元", "预测", "误差"]].to_string(index=False))

print("\n③ 改进：残差随房价变大而变大，说明“总价 = 各项相加”不对路，房价更像“单价 × 面积”")
# 改进一：加交叉特征“区域参考单价 × 面积”（实验 2-3）
ref = (train["总价_万元"] / train["面积"]).groupby(train["区域"]).median()
X_train2 = X_train.assign(参考总价=train["区域"].map(ref) * train["面积"])
X_test2 = X_test.assign(参考总价=test["区域"].map(ref) * test["面积"])
m2 = LinearRegression().fit(X_train2, y_train)
report("加交叉特征（测试集）", y_test, m2.predict(X_test2))
# 改进二：预测 log(总价)，相当于让各因素对总价的影响变成“乘以一个比例”
m3 = LinearRegression().fit(X_train.assign(log面积=np.log(train["面积"])).drop(columns="面积"), np.log(y_train))
pred3 = np.exp(m3.predict(X_test.assign(log面积=np.log(test["面积"])).drop(columns="面积")))
report("预测 log(总价)（测试集）", y_test, pred3)
coef3 = pd.Series(m3.coef_, index=X_train.drop(columns="面积").columns.tolist() + ["log面积"])
print(f"  log 模型的系数：log面积 {coef3['log面积']:.3f}，建成年份 {coef3['建成年份']:.4f}，"
      f"区域_天河 {coef3['区域_天河']:.3f}（即 e^{coef3['区域_天河']:.3f} ≈ {np.exp(coef3['区域_天河']):.2f} 倍），装修_精装 {coef3['装修_精装']:.3f}")
resid3 = y_test - pred3
axes[1].scatter(pred3, resid3, s=6, alpha=0.5, color="#2e7d32")
axes[1].axhline(0, color="red", lw=1)
axes[1].set(xlabel="预测总价（万元）", ylabel="残差（万元）", title="残差图：预测 log(总价)")
plt.tight_layout()
plt.savefig("lab3_2_residual.png", dpi=150)
print(f"  log 模型：预测价 < 200 万的平均绝对残差 {resid3[pred3 < 200].abs().mean():.1f}；> 500 万的 {resid3[pred3 > 500].abs().mean():.1f}")
