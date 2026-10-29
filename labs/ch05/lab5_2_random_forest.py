"""实验 5-2 随机森林预测客户流失

比较逻辑回归、单棵决策树和随机森林，读袋外准确率，输出特征重要性。
运行：python lab5_2_random_forest.py（在 labs/ch05 目录下）
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("../data/telco_churn.csv")
y = (df["是否流失"] == "是").astype(int)
X = pd.get_dummies(df.drop(columns=["客户编号", "是否流失"]),
                   columns=["合约类型", "宽带类型", "技术支持", "视频会员", "付款方式"], dtype=int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集 {len(X_train)} 条，测试集 {len(X_test)} 条，特征 {X.shape[1]} 个，流失率 {y.mean():.1%}")

models = {
    "逻辑回归": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "决策树（不限深度）": DecisionTreeClassifier(random_state=42),
    "决策树（深度 5）": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
    "随机森林（300 棵）": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, oob_score=True,
                                         n_jobs=-1, random_state=42),
}
print(f"\n{'模型':<14}{'训练准确率':>8}{'测试准确率':>8}{'召回率':>6}{'F1':>6}{'AUC':>7}")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    print(f"{name:<14}{model.score(X_train, y_train):>10.3f}{accuracy_score(y_test, pred):>10.3f}"
          f"{recall_score(y_test, pred):>9.3f}{f1_score(y_test, pred):>8.3f}{roc_auc_score(y_test, prob):>8.3f}")

rf = models["随机森林（300 棵）"]
print(f"\n随机森林袋外准确率：{rf.oob_score_:.3f}")

# 树的数量对效果的影响
print(f"\n{'树的数量':<6}{'测试 AUC':>8}")
for n in [1, 5, 10, 50, 100, 300]:
    m = RandomForestClassifier(n_estimators=n, min_samples_leaf=5, n_jobs=-1, random_state=42).fit(X_train, y_train)
    print(f"{n:<8}{roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]):>10.3f}")

# 特征重要性
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n特征重要性（前 8）：")
print(imp.head(8).round(3).to_string())
imp.head(10).iloc[::-1].plot.barh(figsize=(6, 4), color="#3b82c4")
plt.xlabel("重要性")
plt.title("随机森林特征重要性（前 10）")
plt.tight_layout()
plt.savefig("lab5_2_importance.png", dpi=150)

# 流失风险最高的 10 位客户
risk = pd.DataFrame({"客户编号": df.loc[X_test.index, "客户编号"], "流失概率": rf.predict_proba(X_test)[:, 1]})
print("\n流失风险最高的 5 位客户：")
print(risk.sort_values("流失概率", ascending=False).head(5).round(3).to_string(index=False))
