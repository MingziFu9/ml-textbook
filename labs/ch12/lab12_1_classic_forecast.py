"""实验 12-1 门店销量预测（经典方法）

社区超市 2023—2025 年每日销量。用 2025-12-03 及以前的数据，预测 2025-12-04 至 12-31 共 28 天。
① 看数据：趋势、星期效应、节假日；时间序列分解；平稳性检验与差分；自相关图。
② 六种经典方法预测：朴素、季节朴素、移动平均、简单指数平滑、霍尔特—温特斯（Holt-Winters）、SARIMA；
   用 MAE、MAPE 比较。
运行：python lab12_1_classic_forecast.py（在 labs/ch12 目录下）
"""
import logging
import warnings

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")                      # statsmodels 的收敛提示较多，这里不显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("../data/store_sales.csv", parse_dates=["日期"], index_col="日期")
y = df["销量"].asfreq("D")
H = 28
train, test = y[:-H], y[-H:]
print(f"共 {len(y)} 天（{y.index[0]:%Y-%m-%d} 至 {y.index[-1]:%Y-%m-%d}）；训练 {len(train)} 天，"
      f"预测最后 {H} 天（{test.index[0]:%m-%d} 至 {test.index[-1]:%m-%d}）")

# ① 看数据
print("\n① 数据的规律")
print("  各年日均销量：" + "，".join(f"{yr} 年 {v:.0f}" for yr, v in y.groupby(y.index.year).mean().items()))
wd = y[df["节假日"] == "无"].groupby(y[df["节假日"] == "无"].index.dayofweek).mean()
print("  非节假日按星期的日均销量：" + "，".join(f"周{'一二三四五六日'[i]} {v:.0f}" for i, v in wd.items()))
hol = df.groupby("节假日")["销量"].mean()
print("  按节假日的日均销量：" + "，".join(f"{k} {hol[k]:.0f}" for k in ["无", "春节前", "春节", "国庆"]))
promo = df[df["节假日"] == "无"].groupby("是否促销")["销量"].mean()
print(f"  非节假日：不促销 {promo[0]:.0f}，促销 {promo[1]:.0f}（高 {promo[1] / promo[0] - 1:.0%}）")
dec = seasonal_decompose(train, model="multiplicative", period=7)
print("  乘法分解（周期 7）的季节因子：" + "，".join(f"周{'一二三四五六日'[d.dayofweek]} {v:.3f}"
                                      for d, v in dec.seasonal[:7].sort_index(key=lambda i: i.dayofweek).items()))
print("  平稳性检验（ADF 的原假设是“不平稳”，KPSS 的原假设是“平稳”，p < 0.05 表示拒绝原假设）：")
for name, s in [("原序列", train), ("一阶差分", train.diff().dropna())]:
    p_adf = adfuller(s, autolag="AIC")[1]
    p_kpss = kpss(s, regression="c", nlags="auto")[1]          # 查表得到的 p 值只在 0.01～0.1 之间
    print(f"    {name:<6}ADF p = {p_adf:.4f}（{'平稳' if p_adf < 0.05 else '不平稳'}）；"
          f"KPSS p = {p_kpss:.2f}（{'不平稳' if p_kpss < 0.05 else '平稳'}）")
r = acf(train, nlags=28)
print("  自相关系数：滞后 1 天 " + f"{r[1]:.2f}，2 天 {r[2]:.2f}，3 天 {r[3]:.2f}，7 天 {r[7]:.2f}，14 天 {r[14]:.2f}，28 天 {r[28]:.2f}")


# ② 六种方法
def score(pred):
    e = test.values - np.asarray(pred)
    return np.abs(e).mean(), (np.abs(e) / test.values).mean() * 100


preds = {
    "朴素（明天 = 今天）": np.repeat(train.iloc[-1], H),
    "季节朴素（= 上周同一天）": np.tile(train.iloc[-7:].values, H // 7),
    "移动平均（最近 7 天平均）": np.repeat(train.iloc[-7:].mean(), H),
}
ses = SimpleExpSmoothing(train, initialization_method="estimated").fit()
preds["简单指数平滑"] = ses.forecast(H).values
hw = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=7, initialization_method="estimated").fit()
preds["霍尔特—温特斯"] = hw.forecast(H).values
aics = {}
for p_ in range(3):                                    # 在训练集上按 AIC 选阶数（AIC 越小越好），不看测试集
    for q_ in range(2):
        aics[(p_, q_)] = SARIMAX(train, order=(p_, 1, q_), seasonal_order=(0, 1, 1, 7)).fit(disp=False).aic
p_, q_ = min(aics, key=aics.get)
print("\n  SARIMA 选阶：季节部分 (0,1,1,7)，非季节部分 (p,1,q) 的 AIC：" + "，".join(f"({a},1,{b}) {v:.0f}" for (a, b), v in aics.items()))
sar = SARIMAX(train, order=(p_, 1, q_), seasonal_order=(0, 1, 1, 7)).fit(disp=False)
SAR = f"SARIMA({p_},1,{q_})(0,1,1,7)"
preds[SAR] = sar.forecast(H).values
print(f"\n② 预测未来 {H} 天（实际日均 {test.mean():.0f} 件）")
print(f"  简单指数平滑学到的平滑系数 α = {ses.params['smoothing_level']:.3f}；"
      f"霍尔特—温特斯 α = {hw.params['smoothing_level']:.3f}，β = {hw.params['smoothing_trend']:.4f}，γ = {hw.params['smoothing_seasonal']:.3f}")
res = {}
for name, p in preds.items():
    res[name] = score(p)
    print(f"  {name:<24}MAE {res[name][0]:6.1f} 件   MAPE {res[name][1]:5.2f}%")
best = min(res, key=lambda k: res[k][0])
print(f"  MAE 最小：{best}")
pr = test.index[df.loc[test.index, "是否促销"] == 1]
e = (test - preds["霍尔特—温特斯"]).loc[pr]
print(f"  预测期内促销 {len(pr)} 天（{'，'.join(f'{d:%m-%d}' for d in pr)}），霍尔特—温特斯在这几天的误差："
      + "，".join(f"{v:+.0f}" for v in e) + " 件")

fig, ax = plt.subplots(2, 2, figsize=(14, 7.5))
ax[0, 0].plot(y.index, y.values, lw=0.6); ax[0, 0].set_title("三年每日销量（低谷是春节，高峰是春节前和国庆）"); ax[0, 0].set_xlabel("")
tail = slice(-84, None)
ax[0, 1].plot(dec.observed[tail], lw=0.8, label="原序列"); ax[0, 1].plot(dec.trend[tail], lw=2, label="趋势（7 日中心移动平均）")
ax[0, 1].set_title("时间序列分解（训练期最后 12 周）"); ax[0, 1].legend(fontsize=8)
plot_acf(train, lags=28, ax=ax[1, 0]); ax[1, 0].set_title("自相关图：滞后 7、14、21、28 天处突出"); ax[1, 0].set_xlabel("滞后天数")
ax[1, 1].plot(y[-70:].index, y[-70:], color="k", lw=1, label="实际")
for name in ["季节朴素（= 上周同一天）", "霍尔特—温特斯", SAR]:
    ax[1, 1].plot(test.index, preds[name], lw=1.4, label=f"{name} MAE {res[name][0]:.0f}")
ax[1, 1].axvline(test.index[0], color="gray", ls="--"); ax[1, 1].set_title("最后 28 天的预测"); ax[1, 1].legend(fontsize=8)
for a in ax.ravel():
    a.grid(alpha=0.3)
for a in [ax[0, 0], ax[0, 1], ax[1, 1]]:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m" if a is ax[0, 0] else "%m-%d"))
plt.tight_layout(); plt.savefig("lab12_1_forecast.png", dpi=90)
