"""实验 2-3 特征构造、特征选择与数据泄露“侦探”

在清洗好的二手房数据上：① 构造新特征，看线性回归的效果变化；② 用过滤法、包裹法、嵌入法做特征选择；
③ 在一份埋了泄露的数据里找出泄露特征，比较修复前后的成绩。模型统一用线性回归，评价指标是 R²（越接近 1 越好，第 3 章细讲）。
运行：python lab2_3_features_leakage.py（在 labs/ch02 目录下）
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/house_clean.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42)
train, test = train.copy(), test.copy()


def fill(part, ref):
    """缺失值用训练集 ref 的统计量填充（同实验 2-2）"""
    part["建成年份"] = part["建成年份"].fillna(part["区域"].map(ref.groupby("区域")["建成年份"].median()))
    part["距地铁站_米"] = part["距地铁站_米"].fillna(ref["距地铁站_米"].median())
    return part


train, test = fill(train, train), fill(test, train)


def base_features(part):
    X = part[["面积", "建成年份", "距地铁站_米", "总楼层", "室", "厅"]].copy()
    X["楼层位置"] = part["楼层位置"].map({"低": 0, "中": 1, "高": 2})
    return X.join(pd.get_dummies(part[["区域", "装修", "朝向"]], dtype=int))


def r2(X_tr, X_te, model=None):
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    model = LinearRegression() if model is None else model
    model.fit(X_tr, train["总价_万元"])
    return model.score(X_tr, train["总价_万元"]), model.score(X_te, test["总价_万元"])


# ---------- ① 特征构造 ----------
print("① 特征构造（线性回归）")
Xb_tr, Xb_te = base_features(train), base_features(test)
print(f"  原始特征 {Xb_tr.shape[1]} 个：训练 R² {r2(Xb_tr, Xb_te)[0]:.3f}，测试 R² {r2(Xb_tr, Xb_te)[1]:.3f}")

# 区域参考单价：只用训练集计算各区“总价 / 面积”的中位数
ref_price = (train["总价_万元"] / train["面积"]).groupby(train["区域"]).median()


def add_features(part, X):
    X = X.copy()
    listed = pd.to_datetime(part["挂牌日期"])
    X["房龄"] = listed.dt.year - part["建成年份"]                         # 日期特征：挂牌时房子几年了
    X["挂牌月序"] = (listed.dt.year - 2024) * 12 + listed.dt.month       # 日期特征：第几个月挂牌
    X["朝南"] = part["朝向"].isin(["南", "南北"]).astype(int)              # 二值特征
    X["每室面积"] = part["面积"] / part["室"]                              # 比率特征
    X["参考总价"] = part["区域"].map(ref_price) * part["面积"]              # 交叉特征：区域单价水平 × 面积
    return X


Xf_tr, Xf_te = add_features(train, Xb_tr), add_features(test, Xb_te)
for cols, name in [(["房龄", "挂牌月序", "朝南", "每室面积"], "加日期、比率特征"), (list(Xf_tr.columns), "再加交叉特征“参考总价”")]:
    cols = [c for c in Xf_tr.columns if c in Xb_tr.columns or c in cols]
    tr, te = r2(Xf_tr[cols], Xf_te[cols])
    print(f"  {name:<16}{len(cols)} 个：训练 R² {tr:.3f}，测试 R² {te:.3f}")

# ---------- ② 特征选择 ----------
print("\n② 特征选择（在 ① 的全部特征上，各选 10 个）")
scaler = StandardScaler().fit(Xf_tr)
Xs_tr = pd.DataFrame(scaler.transform(Xf_tr), columns=Xf_tr.columns, index=Xf_tr.index)
Xs_te = pd.DataFrame(scaler.transform(Xf_te.reindex(columns=Xf_tr.columns, fill_value=0)), columns=Xf_tr.columns, index=Xf_te.index)
y_tr = train["总价_万元"]

vt = VarianceThreshold(0.01).fit(Xf_tr)
print(f"  方差过滤：方差小于 0.01 的特征 {list(Xf_tr.columns[~vt.get_support()])}")

corr = Xf_tr.corrwith(y_tr).abs().sort_values(ascending=False)
filter_sel = list(corr.index[:10])
print(f"  过滤法（与总价相关系数绝对值最大的 10 个）：{filter_sel}")

rfe = RFE(LinearRegression(), n_features_to_select=10).fit(Xs_tr, y_tr)
wrap_sel = list(Xs_tr.columns[rfe.support_])
print(f"  包裹法（递归特征消除）：{wrap_sel}")

lasso = LassoCV(cv=5, random_state=42).fit(Xs_tr, y_tr)
coef = pd.Series(lasso.coef_, index=Xs_tr.columns)
emb_sel = list(coef.abs().sort_values(ascending=False).index[:10])
print(f"  嵌入法（Lasso，alpha={lasso.alpha_:.3f}，系数为 0 的有 {(coef == 0).sum()} 个）系数最大的 10 个：{emb_sel}")

print("\n  方法          特征数  训练 R²  测试 R²")
for name, cols in [("全部特征", list(Xs_tr.columns)), ("过滤法", filter_sel), ("包裹法", wrap_sel), ("嵌入法", emb_sel)]:
    tr, te = r2(Xs_tr[cols], Xs_te[cols])
    print(f"  {name:<10}{len(cols):>6}{tr:>9.3f}{te:>9.3f}")

# ---------- ③ 数据泄露侦探 ----------
print("\n③ 数据泄露侦探（模型换成随机森林，第 5 章介绍）")
from sklearn.ensemble import RandomForestRegressor

rf = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42)
raw = pd.read_csv("../data/house_raw.csv").drop_duplicates("房源编号").set_index("房源编号")
leak_tr = Xb_tr.assign(单价=raw.loc[train["房源编号"], "单价_元每平方米"].to_numpy())
leak_te = Xb_te.assign(单价=raw.loc[test["房源编号"], "单价_元每平方米"].to_numpy())
tr, te = r2(Xb_tr, Xb_te, rf())
print(f"  原始特征：            训练 R² {tr:.3f}，测试 R² {te:.3f}")
model = rf()
tr, te = r2(leak_tr, leak_te, model)
print(f"  同事给的数据（多了“单价”）：训练 R² {tr:.3f}，测试 R² {te:.3f}")
imp = pd.Series(model.feature_importances_, index=leak_tr.columns).sort_values(ascending=False).head(4).round(3)
print(f"  特征重要性前 4：{imp.to_dict()}")

# 第二种泄露：小区“目标编码”用了全部数据（包括测试集）的总价
print("\n  小区有", df["小区"].nunique(), "个，每个小区平均", round(len(df) / df["小区"].nunique(), 1), "套，独热编码太多，改用“小区参考单价”")


def estate_price(ref, k=5):
    """各小区单价中位数，用所在区的中位数做平滑：挂牌越少的小区越靠近区的水平"""
    unit = ref["总价_万元"] / ref["面积"]
    g = unit.groupby(ref["小区"]).agg(["median", "count"])
    district = unit.groupby(ref["区域"]).median()
    area_of = ref.drop_duplicates("小区").set_index("小区")["区域"]
    return (g["median"] * g["count"] + area_of[g.index].map(district) * k) / (g["count"] + k), district


def with_estate(part, X, ref):
    price, district = estate_price(ref)
    p = part["小区"].map(price).fillna(part["区域"].map(district))   # 训练集里没见过的小区，用所在区的水平
    return X.assign(小区参考总价=p * part["面积"])


for name, ref in [("用全部数据计算（泄露）", df), ("只用训练集计算（正确）", train)]:
    tr, te = r2(with_estate(train, Xb_tr, ref), with_estate(test, Xb_te, ref))
    print(f"  小区参考单价{name}：线性回归 训练 R² {tr:.3f}，测试 R² {te:.3f}")
unseen = (~test["小区"].isin(train["小区"])).sum()
print(f"  测试集 {len(test)} 套中，所在小区在训练集里没出现过的 {unseen} 套")
