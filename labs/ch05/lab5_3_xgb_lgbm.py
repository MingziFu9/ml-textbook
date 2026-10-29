"""实验 5-3 XGBoost 与 LightGBM 对比

在客户流失数据上比较梯度提升树的两种实现：效果、训练时间、学习率与早停；
最后输出测试报告需要的指标和错误案例。
运行：python lab5_3_xgb_lgbm.py（在 labs/ch05 目录下）
"""
import time
import warnings

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# LightGBM 4.7 起提示 eval_set 将改名为 eval_X / eval_y，旧写法仍可用，这里先屏蔽提示
warnings.filterwarnings("ignore", message=".*eval_set.*")

df = pd.read_csv("../data/telco_churn.csv")
y = (df["是否流失"] == "是").astype(int)
X = pd.get_dummies(df.drop(columns=["客户编号", "是否流失"]),
                   columns=["合约类型", "宽带类型", "技术支持", "视频会员", "付款方式"], dtype=int)
# 先分出测试集，再从训练集中分出验证集用于早停（测试集只在最后用一次）
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42, stratify=y_tmp)
print(f"训练集 {len(X_train)}，验证集 {len(X_val)}，测试集 {len(X_test)}")

# 1. 默认参数对比
models = {
    "随机森林": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "sklearn GBDT": GradientBoostingClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
}
print(f"\n{'模型（默认参数）':<16}{'训练 AUC':>8}{'测试 AUC':>8}{'训练用时(秒)':>12}")
for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    cost = time.perf_counter() - start
    tr = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    te = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"{name:<18}{tr:>9.3f}{te:>9.3f}{cost:>12.2f}")

# 2. 学习率与早停：用验证集决定树的数量
print(f"\n{'模型':<10}{'学习率':>6}{'最佳树数':>8}{'验证 AUC':>9}{'测试 AUC':>9}")
for lr in [0.3, 0.1, 0.03]:
    xgb_m = xgb.XGBClassifier(n_estimators=2000, learning_rate=lr, max_depth=3, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="auc", early_stopping_rounds=50,
                              random_state=42, n_jobs=-1)
    xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"{'XGBoost':<12}{lr:>6}{xgb_m.best_iteration + 1:>9}{xgb_m.best_score:>10.3f}"
          f"{roc_auc_score(y_test, xgb_m.predict_proba(X_test)[:, 1]):>9.3f}")
    lgb_m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=lr, num_leaves=8, min_child_samples=30,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=42, verbose=-1)
    lgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    val_auc = lgb_m.best_score_["valid_0"]["auc"]
    print(f"{'LightGBM':<12}{lr:>6}{lgb_m.best_iteration_:>9}{val_auc:>10.3f}"
          f"{roc_auc_score(y_test, lgb_m.predict_proba(X_test)[:, 1]):>9.3f}")

# 3. 测试报告：用学习率 0.03 的 LightGBM 作为最终模型
final = lgb_m
prob = final.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
print("\n===== 测试报告（LightGBM，学习率 0.03，阈值 0.5）=====")
print(f"测试集 AUC：{roc_auc_score(y_test, prob):.3f}")
print("混淆矩阵（行：真实 未流失/流失；列：预测 未流失/流失）：")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["未流失", "流失"], digits=3))

# 4. 错误案例：模型最有把握却判错的客户
cases = df.loc[X_test.index].assign(流失概率=prob.round(3), 预测=pred)
wrong = cases[cases["预测"] != y_test]
cols = ["客户编号", "在网月数", "合约类型", "宽带类型", "月消费", "近半年投诉次数", "是否流失", "流失概率"]
print("漏判（真实流失，模型认为不会流失）概率最低的 5 位：")
print(wrong[wrong["是否流失"] == "是"].nsmallest(5, "流失概率")[cols].to_string(index=False))
print("误判（真实未流失，模型认为会流失）概率最高的 5 位：")
print(wrong[wrong["是否流失"] == "否"].nlargest(5, "流失概率")[cols].to_string(index=False))
