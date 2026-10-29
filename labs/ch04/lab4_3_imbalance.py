"""实验 4-3 类别不平衡：信用卡欺诈识别

2 万笔交易中欺诈只有 1% 左右。比较不处理、类别权重、随机欠采样、SMOTE 过采样四种做法，
分析漏掉的欺诈和误报的正常交易，为模型测试报告准备材料。
运行：python lab4_3_imbalance.py（在 labs/ch04 目录下）；需要 imbalanced-learn：pip install imbalanced-learn
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/card_fraud.csv")
y = df["是否欺诈"]
X = df.drop(columns=["交易编号", "是否欺诈"])
X["交易金额_对数"] = np.log1p(X.pop("交易金额"))          # 金额跨度从几元到几万元，取对数
X["距常用地_对数"] = np.log1p(X.pop("距常用地_公里"))
X["深夜"] = (X["交易小时"] < 6).astype(int)
X = pd.get_dummies(X, columns=["商户类别"], dtype=int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集 {len(y_train)} 笔（欺诈 {y_train.sum()} 笔），测试集 {len(y_test)} 笔（欺诈 {y_test.sum()} 笔），"
      f"欺诈率 {y.mean():.2%}，特征 {X.shape[1]} 个")

print(f"\n① 全部猜“正常”：准确率 {accuracy_score(y_test, np.zeros(len(y_test))):.3f}，一笔欺诈也抓不到")

models = {
    "不处理": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "类别权重 balanced": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "随机欠采样": imb_pipeline(StandardScaler(), RandomUnderSampler(random_state=42), LogisticRegression(max_iter=1000)),
    "SMOTE 过采样": imb_pipeline(StandardScaler(), SMOTE(random_state=42), LogisticRegression(max_iter=1000)),
}
print("\n② 四种做法（测试集，阈值 0.5）")
print(f"  {'做法':<16}{'TP':>4}{'FN':>4}{'FP':>5}{'TN':>6}  准确率  精确率  召回率    F1    AUC     AP")
probs = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    prob = m.predict_proba(X_test)[:, 1]
    probs[name] = prob
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"  {name:<14}{tp:>5}{fn:>4}{fp:>5}{tn:>6}  {accuracy_score(y_test, pred):.3f}  {precision_score(y_test, pred):.3f}"
          f"  {recall_score(y_test, pred):.3f}  {f1_score(y_test, pred):.3f}  {roc_auc_score(y_test, prob):.3f}"
          f"  {average_precision_score(y_test, prob):.3f}")

print("\n③ 不处理的模型只调阈值：")
print(f"  {'阈值':<6}{'TP':>4}{'FN':>4}{'FP':>5}  精确率  召回率    F1")
for t in [0.5, 0.3, 0.2, 0.1, 0.05]:
    pred = (probs["不处理"] >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"  {t:<6}{tp:>4}{fn:>4}{fp:>5}  {precision_score(y_test, pred):.3f}  {recall_score(y_test, pred):.3f}  {f1_score(y_test, pred):.3f}")

# ④ 错误案例分析：以“类别权重”模型为例
prob = probs["类别权重 balanced"]
test = df.loc[X_test.index].assign(欺诈概率=prob.round(3))
pred = prob >= 0.5
cols = ["交易编号", "交易金额", "交易小时", "商户类别", "是否境外", "是否新设备", "距上次交易_小时", "近24小时交易次数", "距常用地_公里", "欺诈概率"]
fn_cases = test[(test["是否欺诈"] == 1) & ~pred].sort_values("欺诈概率")
fp_cases = test[(test["是否欺诈"] == 0) & pred].sort_values("欺诈概率", ascending=False)
print(f"\n④ 错误案例分析（类别权重模型）：漏判 {len(fn_cases)} 笔欺诈，误报 {len(fp_cases)} 笔正常交易")
print("  漏判的欺诈交易：")
print(fn_cases[cols].to_string(index=False))
print("  欺诈概率最高的 5 笔误报：")
print(fp_cases[cols].head(5).to_string(index=False))
num = ["交易金额", "交易小时", "是否境外", "是否新设备", "距上次交易_小时", "近24小时交易次数", "距常用地_公里"]
summary = pd.DataFrame({"测试集欺诈（中位数/比例）": test[test["是否欺诈"] == 1][num].median(),
                        "漏判的欺诈": fn_cases[num].median(),
                        "测试集正常交易": test[test["是否欺诈"] == 0][num].median(),
                        "误报的正常交易": fp_cases[num].median()})
for c in ["是否境外", "是否新设备"]:   # 0/1 特征用比例代替中位数
    summary.loc[c] = [test[test["是否欺诈"] == 1][c].mean(), fn_cases[c].mean(),
                      test[test["是否欺诈"] == 0][c].mean(), fp_cases[c].mean()]
print("  各组特征对比（数值特征取中位数，是否境外、是否新设备取比例）：")
print(summary.round(2).to_string())

# ⑤ 业务代价：漏掉一笔欺诈，损失这笔交易的金额；误报一笔正常交易，要人工打电话核实，按每笔 20 元计
amount = df.loc[X_test.index, "交易金额"].to_numpy()
print("\n⑤ 业务代价（测试集 6000 笔；漏判损失 = 漏掉的欺诈金额之和，误报成本 = 误报笔数 × 20 元）")
print(f"  {'做法':<22}{'漏判':>4}{'误报':>6}{'漏判损失':>10}{'误报成本':>10}{'总代价':>10}")
plans = [(name, probs[name], 0.5) for name in models] + [("不处理 + 阈值 0.1", probs["不处理"], 0.1),
                                                         ("类别权重 + 阈值 0.8", probs["类别权重 balanced"], 0.8)]
for name, p, t in plans:
    pred = p >= t
    miss = (y_test.to_numpy() == 1) & ~pred
    false_alarm = (y_test.to_numpy() == 0) & pred
    loss, cost = amount[miss].sum(), false_alarm.sum() * 20
    print(f"  {name:<20}{miss.sum():>5}{false_alarm.sum():>7}{loss:>11.0f}{cost:>10.0f}{loss + cost:>11.0f}")
print(f"  什么都不做（全部放行）：漏判 {y_test.sum()} 笔，损失 {amount[y_test.to_numpy() == 1].sum():.0f} 元")
