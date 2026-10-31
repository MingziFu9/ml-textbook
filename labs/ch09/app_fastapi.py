"""实验 9-3 第 1 部分：用 FastAPI 把流失预测模型做成网络接口

先运行 lab9_2_pipeline_tuning.py 生成 churn_model.joblib，再在 labs/ch09 目录下启动：
    uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
浏览器打开 http://127.0.0.1:8000/docs 可以在网页上直接试用接口。
"""
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

model = joblib.load("churn_model.joblib")          # 启动时加载一次，不要每次请求都加载
app = FastAPI(title="客户流失预测接口", version="1.0")


class Customer(BaseModel):
    """一位客户的信息；字段名、取值范围与训练数据一致，不合法的请求会被自动拒绝"""
    年龄: int = Field(ge=18, le=100)
    在网月数: int = Field(ge=0, le=120)
    合约类型: Literal["月付", "一年", "两年"]
    宽带类型: Literal["光纤", "DSL", "无"]
    技术支持: Literal["是", "否", "无网络"]
    视频会员: Literal["是", "否", "无网络"]
    付款方式: Literal["自动扣款", "手机支付", "营业厅缴费"]
    月消费: float
    累计消费: float
    近半年投诉次数: int = Field(ge=0, le=20)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(c: Customer):
    prob = float(model.predict_proba(pd.DataFrame([c.model_dump()]))[0, 1])
    level = "高" if prob >= 0.6 else "中" if prob >= 0.3 else "低"
    return {"流失概率": round(prob, 3), "风险等级": level}
