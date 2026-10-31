"""实验 9-3 把模型发布成服务，并检查数据漂移

① 加载保存的模型，确认和保存前的预测一致。
② 用 FastAPI 的测试客户端调用接口（不用真的启动服务器），包括一个不合法的请求。
③ 调用 Gradio 小工具的预测函数。
④ 数据漂移：模拟“三个月后”的新客户（涨价后月消费整体上升、月付合约变多），用 PSI 指标检查哪些特征漂移了。
运行：先运行 lab9_2_pipeline_tuning.py，再 python lab9_3_serve_monitor.py（在 labs/ch09 目录下）
"""
import time

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from app_fastapi import app
from app_gradio import predict

df = pd.read_csv("../data/telco_churn.csv")
X = df.drop(columns=["客户编号", "是否流失"])
y = (df["是否流失"] == "是").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ① 加载模型
t = time.perf_counter()
model = joblib.load("churn_model.joblib")
print(f"① 加载模型用时 {time.perf_counter() - t:.3f} 秒；测试集前 3 位客户的流失概率："
      + "，".join(f"{p:.3f}" for p in model.predict_proba(X_test.head(3))[:, 1]))

# ② FastAPI 接口
client = TestClient(app)
print(f"\n② GET /health → {client.get('/health').json()}")
c = X_test.iloc[0].to_dict()
c = {k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()}
r = client.post("/predict", json=c)
print(f"  POST /predict，请求：{c}")
print(f"  状态码 {r.status_code}，返回：{r.json()}")
bad = dict(c, 合约类型="三年", 年龄=8)
r = client.post("/predict", json=bad)
print(f"  不合法的请求（合约类型=三年，年龄=8）→ 状态码 {r.status_code}，出错字段："
      + "，".join(e["loc"][-1] for e in r.json()["detail"]))
t = time.perf_counter()
for _ in range(100):
    client.post("/predict", json=c)
print(f"  连续请求 100 次，平均每次 {(time.perf_counter() - t) * 10:.1f} 毫秒")

# ③ Gradio
print(f"\n③ Gradio 预测函数：月付、在网 2 个月、投诉 2 次 → {predict(38, 2, '月付', '光纤', '否', '是', '自动扣款', 97.8, 192.8, 2)}")
print(f"  两年合约、在网 60 个月、无投诉 → {predict(45, 60, '两年', 'DSL', '是', '否', '自动扣款', 55.0, 3300.0, 0)}")


# ④ 数据漂移：PSI（群体稳定性指数）
def psi(old, new, bins=10):
    """按旧数据的十分位数分箱，比较新旧两批数据各箱的占比"""
    if old.dtype == object or str(old.dtype) in ("str", "string", "category"):
        cats = old.unique()
        p = old.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        q = new.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    else:
        edges = np.unique(np.quantile(old, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(old, edges)[0] / len(old)
        q = np.histogram(new, edges)[0] / len(new)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(((q - p) * np.log(q / p)).sum())


rng = np.random.default_rng(7)
new = X_test.copy()
new["月消费"] = (new["月消费"] * 1.15 + rng.normal(0, 3, len(new))).round(1)          # 涨价 15%
flip = (new["合约类型"] != "月付") & (rng.random(len(new)) < 0.4)                     # 四成长期合约客户改为月付
new.loc[flip, "合约类型"] = "月付"
print("\n④ 数据漂移检查（PSI < 0.1 稳定，0.1～0.25 轻微漂移，> 0.25 明显漂移）")
for col in X.columns:
    v = psi(X_train[col], new[col])
    print(f"  {col:<8} PSI {v:.3f}  {'明显漂移' if v > 0.25 else '轻微漂移' if v > 0.1 else '稳定'}")
p_old, p_new = model.predict_proba(X_test)[:, 1].mean(), model.predict_proba(new)[:, 1].mean()
print(f"  模型给出的平均流失概率：原测试集 {p_old:.3f} → 新数据 {p_new:.3f}")
