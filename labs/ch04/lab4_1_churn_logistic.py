"""实验 4-1 电信客户流失预测

用逻辑回归预测客户是否流失：看混淆矩阵和各项指标，解读系数，输出流失概率排名前 50 的客户名单。
划分方式与实验 5-2 相同（7 : 3 分层，random_state=42），结果可以直接对比。
运行：python lab4_1_churn_logistic.py（在 labs/ch04 目录下）
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/telco_churn.csv")
y = (df["是否流失"] == "是").astype(int)
X = pd.get_dummies(df.drop(columns=["客户编号", "是否流失"]),
                   columns=["合约类型", "宽带类型", "技术支持", "视频会员", "付款方式"], dtype=int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集 {len(X_train)} 条，测试集 {len(X_test)} 条，特征 {X.shape[1]} 个；"
      f"训练集流失率 {y_train.mean():.1%}，测试集流失率 {y_test.mean():.1%}")

# ① 基线：所有客户都猜“不流失”
base = np.zeros(len(y_test), dtype=int)
print(f"\n① 基线（全猜不流失）：准确率 {accuracy_score(y_test, base):.3f}，召回率 {recall_score(y_test, base):.3f}")

# ② 逻辑回归
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]          # 每个客户流失的概率
pred = model.predict(X_test)                      # 默认阈值 0.5：概率 ≥ 0.5 判为流失
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print("\n② 逻辑回归（阈值 0.5）")
print(f"  混淆矩阵：TP={tp}  FN={fn}  FP={fp}  TN={tn}")
print(f"  训练准确率 {model.score(X_train, y_train):.3f}  测试准确率 {accuracy_score(y_test, pred):.3f}")
print(f"  精确率 {precision_score(y_test, pred):.3f}  召回率 {recall_score(y_test, pred):.3f}  "
      f"F1 {f1_score(y_test, pred):.3f}  AUC {roc_auc_score(y_test, prob):.3f}")

# ③ 系数解读：特征已标准化，系数可以比较大小；e^系数 是“几率”变成原来的几倍
lr = model[-1]
coef = pd.Series(lr.coef_[0], index=X.columns).sort_values()
print(f"\n③ 截距 {lr.intercept_[0]:.3f}；系数（特征已标准化，按大小排序）：")
for name in list(coef.index[:5]) + list(coef.index[-5:]):
    print(f"  {name:<12}{coef[name]:>8.3f}   e^系数 = {np.exp(coef[name]):.2f}")

# ④ 手算一个客户的流失概率
i = 0
z_std = model[0].transform(X_test.iloc[[i]])[0]
z = lr.intercept_[0] + (lr.coef_[0] * z_std).sum()
print(f"\n④ 测试集第 1 个客户 {df.loc[X_test.index[i], '客户编号']}：z = {z:.3f}，"
      f"Sigmoid(z) = 1 / (1 + e^{-z:.3f}) = {1 / (1 + np.exp(-z)):.3f}；predict_proba 给出 {prob[i]:.3f}")

# ⑤ 流失概率最高的 50 个客户
top = df.loc[X_test.index].assign(流失概率=prob.round(3)).sort_values("流失概率", ascending=False).head(50)
top[["客户编号", "在网月数", "合约类型", "月消费", "近半年投诉次数", "流失概率", "是否流失"]].to_csv(
    "churn_top50.csv", index=False, encoding="utf-8-sig")
hit = (top["是否流失"] == "是").sum()
print(f"\n⑤ 流失概率前 50 名中，实际流失 {hit} 人（命中率 {hit / 50:.0%}）；随机抽 50 人预计只有 {y_test.mean() * 50:.0f} 人")
print(top[["客户编号", "在网月数", "合约类型", "月消费", "近半年投诉次数", "流失概率", "是否流失"]].head(8).to_string(index=False))
print(f"  前 50 名的流失概率在 {top['流失概率'].min():.3f}～{top['流失概率'].max():.3f} 之间；名单已保存到 churn_top50.csv")
