"""实验 8-1 电商客户分群（RFM 模型）

2000 个客户一年的 11605 条订单：计算每个客户的 R（最近一次购买距今天数）、F（购买次数）、M（消费总额），
对数变换、标准化后用 K-Means 分群，用肘部法则和轮廓系数定 K，描述每群客户并起名，导出客户分群名单。
运行：python lab8_1_rfm_kmeans.py（在 labs/ch08 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.width", 200)

orders = pd.read_csv("../data/orders.csv", parse_dates=["下单日期"])
today = pd.Timestamp("2026-10-01")                    # 分析日：数据截止日的第二天
print(f"订单 {len(orders)} 条，客户 {orders['客户编号'].nunique()} 个，"
      f"日期 {orders['下单日期'].min():%Y-%m-%d} 至 {orders['下单日期'].max():%Y-%m-%d}")

# ① 计算 RFM
rfm = orders.groupby("客户编号").agg(
    R=("下单日期", lambda d: (today - d.max()).days),
    F=("订单编号", "count"),
    M=("实付金额", "sum"))
rfm["M"] = rfm["M"].round(1)
c = rfm.index[0]
sub = orders[orders["客户编号"] == c]
print(f"\n① 例：客户 {c} 共下单 {len(sub)} 次，最近一次 {sub['下单日期'].max():%Y-%m-%d}，合计 {sub['实付金额'].sum():.1f} 元"
      f" → R = {rfm.loc[c, 'R']}，F = {rfm.loc[c, 'F']}，M = {rfm.loc[c, 'M']}")
print(rfm.describe().round(1).loc[["mean", "50%", "min", "max"]].to_string())
print(f"  M 的均值 {rfm['M'].mean():.0f} 远大于中位数 {rfm['M'].median():.0f}，右偏严重；"
      f"偏度：R {rfm['R'].skew():.2f}，F {rfm['F'].skew():.2f}，M {rfm['M'].skew():.2f}")

# ② 对数变换 + 标准化
Z = StandardScaler().fit_transform(np.log1p(rfm[["R", "F", "M"]]))
print(f"\n② 取 log(1 + x) 后的偏度：" + "，".join(f"{k} {v:.2f}" for k, v in
                                       np.log1p(rfm[["R", "F", "M"]]).skew().items()) + "；再标准化")

# ③ 肘部法则和轮廓系数
print("\n③ K 从 2 到 8")
print("   K   簇内平方和   轮廓系数")
ks, sse, sil = range(2, 9), [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    sse.append(km.inertia_)
    sil.append(silhouette_score(Z, km.labels_))
    print(f"  {k:>2}   {km.inertia_:>9.1f}   {sil[-1]:.3f}")
best = list(ks)[int(np.argmax(sil))]
print(f"  轮廓系数最高的 K = {best}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(list(ks), sse, "o-"); ax[0].set_title("肘部法则：簇内平方和"); ax[0].set_xlabel("K")
ax[1].plot(list(ks), sil, "s-", color="C1"); ax[1].set_title("轮廓系数"); ax[1].set_xlabel("K")
for a in ax:
    a.axvline(best, color="gray", ls="--", lw=1)
plt.tight_layout()
plt.savefig("lab8_1_k.png", dpi=120)

# ④ K = 4 分群，描述每一群
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Z)
rfm["簇"] = km.labels_
prof = rfm.groupby("簇").agg(人数=("R", "size"), R中位数=("R", "median"), F中位数=("F", "median"),
                            M中位数=("M", "median"), M合计=("M", "sum"))
prof["人数占比"] = (prof["人数"] / len(rfm)).round(3)
prof["消费额占比"] = (prof["M合计"] / rfm["M"].sum()).round(3)
# 按特点起名：消费最高的是高价值；其余中最久没来的是流失风险；剩下两群中消费低的是低价值沉睡，另一群是新客与潜力客户
names = {}
rest = list(prof.index)
hi = prof.loc[rest, "M中位数"].idxmax(); names[hi] = "高价值忠实客户"; rest.remove(hi)
risk = prof.loc[rest, "R中位数"].idxmax(); names[risk] = "流失风险客户"; rest.remove(risk)
low = prof.loc[rest, "M中位数"].idxmin(); names[low] = "低价值沉睡客户"; rest.remove(low)
names[rest[0]] = "新客与潜力客户"
prof["名称"] = prof.index.map(names)
rfm["客户群"] = rfm["簇"].map(names)
print("\n④ K = 4 时各群的特点（R 越小越近）")
print(prof[["名称", "人数", "人数占比", "R中位数", "F中位数", "M中位数", "消费额占比"]].sort_values("M中位数", ascending=False).to_string())
print(f"  轮廓系数 {silhouette_score(Z, km.labels_):.3f}")

colors = {"高价值忠实客户": "C3", "流失风险客户": "C1", "新客与潜力客户": "C2", "低价值沉睡客户": "C7"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, g in rfm.groupby("客户群"):
    ax[0].scatter(g["R"], g["F"], s=6, c=colors[name], label=name, alpha=0.6)
    ax[1].scatter(g["F"], g["M"], s=6, c=colors[name], label=name, alpha=0.6)
ax[0].set_xlabel("R：最近一次购买距今（天）"), ax[0].set_ylabel("F：购买次数")
ax[1].set_xlabel("F：购买次数"), ax[1].set_ylabel("M：消费总额（元）"), ax[1].set_yscale("log")
ax[0].legend(fontsize=8, markerscale=2)
plt.tight_layout()
plt.savefig("lab8_1_segments.png", dpi=120)

# ⑤ 导出名单
rfm.drop(columns="簇").to_csv("customer_segments.csv", encoding="utf-8-sig")
print("\n⑤ 每群各举 2 个客户：")
print(rfm.groupby("客户群").head(2).sort_values("客户群")[["R", "F", "M", "客户群"]].to_string())
print("  完整名单已保存到 customer_segments.csv")
