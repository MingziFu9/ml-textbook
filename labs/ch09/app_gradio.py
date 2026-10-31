"""实验 9-3 第 2 部分：用 Gradio 做一个网页小工具

在 labs/ch09 目录下运行：python app_gradio.py，浏览器打开 http://127.0.0.1:7860 ；
把 launch() 改成 launch(share=True)，会得到一个临时公网链接，可以发给同学试用（最长保留约一周）。
"""
import gradio as gr
import joblib
import pandas as pd

model = joblib.load("churn_model.joblib")


def predict(年龄, 在网月数, 合约类型, 宽带类型, 技术支持, 视频会员, 付款方式, 月消费, 累计消费, 近半年投诉次数):
    row = pd.DataFrame([locals()])
    prob = float(model.predict_proba(row)[0, 1])
    return {"会流失": round(prob, 3), "不会流失": round(1 - prob, 3)}


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Slider(18, 75, 35, step=1, label="年龄"), gr.Slider(1, 72, 12, step=1, label="在网月数"),
            gr.Radio(["月付", "一年", "两年"], value="月付", label="合约类型"),
            gr.Radio(["光纤", "DSL", "无"], value="光纤", label="宽带类型"),
            gr.Radio(["是", "否", "无网络"], value="否", label="技术支持"),
            gr.Radio(["是", "否", "无网络"], value="否", label="视频会员"),
            gr.Radio(["自动扣款", "手机支付", "营业厅缴费"], value="自动扣款", label="付款方式"),
            gr.Number(70, label="月消费（元）"), gr.Number(800, label="累计消费（元）"),
            gr.Slider(0, 5, 0, step=1, label="近半年投诉次数")],
    outputs=gr.Label(label="预测结果"),
    title="客户流失风险预测",
    description="输入客户信息，模型给出流失概率（实验 9-2 训练的梯度提升树）。",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
