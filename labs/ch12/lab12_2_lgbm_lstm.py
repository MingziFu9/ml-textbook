"""实验 12-2 机器学习与 LSTM 时序预测对比

同实验 12-1：用 2025-12-03 及以前的数据，预测 2025-12-04 至 12-31 共 28 天。
① 构造特征：日期特征、促销与节假日（提前知道）、滞后特征（28、35、42、364 天前）、滑动窗口统计量；
② LightGBM：先用时间序列交叉验证评估（并演示打乱顺序的 K 折为什么会高估），再预测最后 28 天；
③ LSTM：用过去 56 天的销量，一次输出未来 28 天；
④ 与实验 12-1 的霍尔特—温特斯比较。
运行：python lab12_2_lgbm_lstm.py（在 labs/ch12 目录下），约 1 分钟
"""
import logging
import time
import warnings

import lightgbm as lgb
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from torch import nn

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
torch.set_num_threads(4)

df = pd.read_csv("../data/store_sales.csv", parse_dates=["日期"], index_col="日期").asfreq("D")
H = 28
y = df["销量"]
test_idx = y.index[-H:]

# ① 特征：预测 28 天以后的销量，只能用至少 28 天以前的销量
X = pd.DataFrame(index=df.index)
X["星期"] = df.index.dayofweek + 1
X["月份"] = df.index.month
X["年内第几天"] = df.index.dayofyear
X["是否促销"] = df["是否促销"]
X["节假日"] = df["节假日"].map({"无": 0, "春节前": 1, "春节": 2, "国庆": 3})
for lag in [28, 35, 42, 364]:
    X[f"{lag}天前销量"] = y.shift(lag)
X["28天前起7日均值"] = y.shift(28).rolling(7).mean()              # 滑动窗口：第 28～34 天前的平均
X["28天前起28日均值"] = y.shift(28).rolling(28).mean()
data = X.join(y).dropna()
train = data.loc[: test_idx[0] - pd.Timedelta(days=1)]
test = data.loc[test_idx]
feats = list(X.columns)
print(f"① 特征 {len(feats)} 个：{'、'.join(feats)}")
print(f"  有完整滞后特征的训练样本 {len(train)} 天（{train.index[0]:%Y-%m-%d} 起），测试 {len(test)} 天")
print("  训练集前 2 行（只显示部分列）：")
print(train[["星期", "是否促销", "28天前销量", "364天前销量", "28天前起7日均值", "销量"]].head(2).round(1).to_string())

# ② LightGBM
params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10, verbose=-1, random_state=0)
mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))          # noqa: E731
mape = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b)) / np.asarray(a)) * 100)  # noqa: E731
print("\n② LightGBM")
tss = TimeSeriesSplit(n_splits=5, test_size=H)
cv_lgb, cv_naive = [], []
for k, (tr, va) in enumerate(tss.split(train), 1):
    m = lgb.LGBMRegressor(**params).fit(train.iloc[tr][feats], train.iloc[tr]["销量"])
    cv_lgb.append(mae(train.iloc[va]["销量"], m.predict(train.iloc[va][feats])))
    cv_naive.append(mae(train.iloc[va]["销量"], train.iloc[va]["28天前销量"]))
    print(f"  时间序列交叉验证第 {k} 折：训练到 {train.index[tr[-1]]:%Y-%m-%d}，验证 {train.index[va[0]]:%m-%d}～{train.index[va[-1]]:%m-%d}，"
          f"LightGBM MAE {cv_lgb[-1]:.1f}，“= 4 周前同一天” MAE {cv_naive[-1]:.1f}")
print(f"  平均：LightGBM {np.mean(cv_lgb):.1f}，4 周前同一天 {np.mean(cv_naive):.1f}")
kf = []
for tr, va in KFold(5, shuffle=True, random_state=0).split(train):
    m = lgb.LGBMRegressor(**params).fit(train.iloc[tr][feats], train.iloc[tr]["销量"])
    kf.append(mae(train.iloc[va]["销量"], m.predict(train.iloc[va][feats])))
print(f"  对比：打乱顺序的 5 折交叉验证 MAE {np.mean(kf):.1f}（用了“未来”的数据训练，偏乐观）")
t = time.perf_counter()
gbm = lgb.LGBMRegressor(**params).fit(train[feats], train["销量"])
sec_lgb = time.perf_counter() - t
p_lgb = gbm.predict(test[feats])
imp = pd.Series(gbm.booster_.feature_importance("gain"), index=feats)
imp = (imp / imp.sum()).sort_values(ascending=False)
print("  特征重要性（增益占比）前 5：" + "，".join(f"{k} {v:.1%}" for k, v in imp.head(5).items()))
print(f"  测试 28 天：MAE {mae(test['销量'], p_lgb):.1f}，MAPE {mape(test['销量'], p_lgb):.2f}%，训练用时 {sec_lgb:.2f} 秒")
pr = test.index[test["是否促销"] == 1]
print("  促销日的误差：" + "，".join(f"{d:%m-%d} {test.loc[d, '销量'] - p:+.0f}" for d, p in zip(test.index, p_lgb) if d in pr) + " 件")

# ③ LSTM：输入过去 56 天，输出未来 28 天
W = 56
series = y.loc[: test_idx[0] - pd.Timedelta(days=1)].values.astype("float32")
mu, sd = series.mean(), series.std()
z = (series - mu) / sd                                              # 标准化，只用训练期的均值和标准差
Xs = np.stack([z[i:i + W] for i in range(len(z) - W - H + 1)])
Ys = np.stack([z[i + W:i + W + H] for i in range(len(z) - W - H + 1)])
print(f"\n③ LSTM：滑动窗口切出 {len(Xs)} 个样本，每个输入 {W} 天、输出 {H} 天")
n_val = 4 * H                                                     # 最后 112 个窗口做验证，用于早停
Xt, Yt = torch.tensor(Xs[:-n_val, :, None]), torch.tensor(Ys[:-n_val])
Xv, Yv = torch.tensor(Xs[-n_val:, :, None]), torch.tensor(Ys[-n_val:])


class Seq2Out(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.out = nn.Linear(64, H)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h[:, -1])                                  # 读完 56 天后的隐藏状态 → 28 个数


torch.manual_seed(0)
net = Seq2Out()
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
best, best_state, wait, log = 1e9, None, 0, []
t = time.perf_counter()
for ep in range(1, 301):
    net.train()
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 32):
        b = perm[i:i + 32]
        loss = nn.functional.mse_loss(net(Xt[b]), Yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        vl = nn.functional.mse_loss(net(Xv), Yv).item()
    log.append(vl)
    if vl < best - 1e-4:
        best, best_state, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= 20:
            break
net.load_state_dict(best_state)
sec_lstm = time.perf_counter() - t
print(f"  训练 {len(log)} 轮后早停，验证损失最低在第 {int(np.argmin(log)) + 1} 轮（{min(log):.3f}），用时 {sec_lstm:.1f} 秒；"
      f"参数 {sum(p.numel() for p in net.parameters())}")
with torch.no_grad():
    p_lstm = net(torch.tensor(z[-W:].reshape(1, W, 1))).numpy().ravel() * sd + mu
print(f"  测试 28 天：MAE {mae(test['销量'], p_lstm):.1f}，MAPE {mape(test['销量'], p_lstm):.2f}%")

# ④ 汇总
hw = ExponentialSmoothing(y.loc[: test_idx[0] - pd.Timedelta(days=1)], trend="add", seasonal="mul", seasonal_periods=7,
                          initialization_method="estimated").fit()
p_hw = hw.forecast(H).values
print("\n④ 最后 28 天的预测对比")
for name, p in [("季节朴素（实验 12-1）", np.tile(y.loc[: test_idx[0] - pd.Timedelta(days=1)].values[-7:], 4)),
                ("霍尔特—温特斯（实验 12-1）", p_hw), ("LightGBM", p_lgb), ("LSTM", p_lstm)]:
    print(f"  {name:<20}MAE {mae(test['销量'], p):6.1f}   MAPE {mape(test['销量'], p):5.2f}%")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].bar(np.arange(1, 6) - 0.2, cv_lgb, 0.4, label="LightGBM"); ax[0].bar(np.arange(1, 6) + 0.2, cv_naive, 0.4, label="4 周前同一天")
ax[0].axhline(np.mean(kf), color="r", ls="--", label=f"打乱的 K 折 {np.mean(kf):.0f}")
ax[0].set_xlabel("时间序列交叉验证的折"); ax[0].set_ylabel("MAE（件）"); ax[0].set_title("交叉验证 MAE"); ax[0].legend(fontsize=8)
imp.head(8)[::-1].plot.barh(ax=ax[1]); ax[1].set_title("LightGBM 特征重要性（增益占比）")
ax[2].plot(test.index, test["销量"], "k-o", ms=3, label="实际")
for name, p in [("霍尔特—温特斯", p_hw), ("LightGBM", p_lgb), ("LSTM", p_lstm)]:
    ax[2].plot(test.index, p, lw=1.3, label=f"{name} MAE {mae(test['销量'], p):.0f}")
for d in pr:
    ax[2].axvline(d, color="orange", alpha=0.4, lw=6)
ax[2].set_title("最后 28 天（橙色竖条是促销日）"); ax[2].legend(fontsize=8)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("lab12_2_compare.png", dpi=90)
