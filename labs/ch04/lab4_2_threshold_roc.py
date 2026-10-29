"""实验 4-2 阈值、混淆矩阵与 ROC

乳腺肿瘤诊断数据（scikit-learn 自带，569 例）：训练逻辑回归，调整分类阈值，观察精确率与召回率此消彼长，
画 ROC 曲线和 PR 曲线，按“漏诊代价高”选阈值。正类（1）= 恶性。
运行：python lab4_2_threshold_roc.py（在 labs/ch04 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

data = load_breast_cancer()
X, y = data.data, 1 - data.target          # 原数据 0 = 恶性、1 = 良性；这里反过来，让“恶性”为正类 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"共 {len(y)} 例，恶性 {y.sum()} 例（{y.mean():.1%}），特征 {X.shape[1]} 个；"
      f"训练集 {len(y_train)} 例，测试集 {len(y_test)} 例（恶性 {y_test.sum()} 例）")

# 为了让阈值的影响更明显，只用 5 个特征（全部 30 个特征见思考题）
cols = [list(data.feature_names).index(c) for c in
        ["mean radius", "mean texture", "mean smoothness", "mean symmetry", "mean fractal dimension"]]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train[:, cols], y_train)
prob = model.predict_proba(X_test[:, cols])[:, 1]

print("\n① 不同阈值下的混淆矩阵和指标（测试集）")
print("  阈值   TP  FN  FP  TN   准确率  精确率  召回率    F1")
rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    acc = (tp + tn) / len(y_test)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r) if p + r else 0
    rows.append((t, tp, fn, fp, tn, acc, p, r, f1))
    print(f"  {t:.1f}  {tp:>4}{fn:>4}{fp:>4}{tn:>4}  {acc:>7.3f}{p:>8.3f}{r:>8.3f}{f1:>7.3f}")

auc = roc_auc_score(y_test, prob)
ap = average_precision_score(y_test, prob)
print(f"\n② AUC = {auc:.3f}；PR 曲线下面积（平均精确率 AP）= {ap:.3f}；恶性比例 {y_test.mean():.3f}（PR 曲线的随机基线）")

# ③ 按“召回率至少 0.95”选阈值：在满足条件的阈值中选最大的一个，让误报尽量少
fpr, tpr, thr = roc_curve(y_test, prob)
ok = tpr >= 0.95
t_best = thr[ok][0]
pred = (prob >= t_best).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"\n③ 要求召回率 ≥ 0.95：阈值取 {t_best:.3f}，TP={tp} FN={fn} FP={fp} TN={tn}，"
      f"召回率 {tp / (tp + fn):.3f}，精确率 {tp / (tp + fp):.3f}，假阳性率 {fp / (fp + tn):.3f}")
print("  注意：阈值是在测试集上挑的，结果偏乐观；正式做法是在验证集上挑阈值，再到测试集上检验（第 9 章）")

# ④ 全部 30 个特征
full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
prob30 = full.predict_proba(X_test)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_test, (prob30 >= 0.5).astype(int)).ravel()
print(f"\n④ 用全部 30 个特征：AUC = {roc_auc_score(y_test, prob30):.3f}，AP = {average_precision_score(y_test, prob30):.3f}，阈值 0.5 时 TP={tp} FN={fn} FP={fp} TN={tn}")

# 画 ROC 和 PR 曲线
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(fpr, tpr, label=f"5 个特征（AUC = {auc:.3f}）")
f30, t30, _ = roc_curve(y_test, prob30)
axes[0].plot(f30, t30, label=f"30 个特征（AUC = {roc_auc_score(y_test, prob30):.3f}）")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="随机猜（AUC = 0.5）")
for t, *_rest in rows:
    if t in (0.2, 0.5, 0.8):
        pr_ = (prob >= t)
        x_ = (pr_ & (y_test == 0)).sum() / (y_test == 0).sum()
        y_ = (pr_ & (y_test == 1)).sum() / (y_test == 1).sum()
        axes[0].annotate(f"阈值 {t}", (x_, y_), xytext=(x_ + 0.08, y_ - 0.12), arrowprops={"arrowstyle": "->"})
        axes[0].plot(x_, y_, "o", color="#e53935")
axes[0].set(xlabel="假阳性率 FPR（良性被误判为恶性的比例）", ylabel="真阳性率 TPR（召回率）", title="ROC 曲线")
axes[0].legend(loc="lower right")
prec, rec, _ = precision_recall_curve(y_test, prob)
axes[1].plot(rec, prec, label=f"5 个特征（AP = {ap:.3f}）")
p30, r30, _ = precision_recall_curve(y_test, prob30)
axes[1].plot(r30, p30, label=f"30 个特征（AP = {average_precision_score(y_test, prob30):.3f}）")
axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"随机猜（{y_test.mean():.3f}）")
axes[1].set(xlabel="召回率", ylabel="精确率", title="PR 曲线", ylim=(0, 1.05))
axes[1].legend(loc="lower left")
plt.tight_layout()
plt.savefig("lab4_2_roc_pr.png", dpi=150)

# 导出 5 个特征模型在测试集上的概率，给交互动画“阈值与混淆矩阵”使用
import json
json.dump({"prob": [round(float(p), 4) for p in prob], "y": [int(v) for v in y_test]},
          open("lab4_2_scores.json", "w"))
