"""实验 9-2 一条完整的建模流水线

① 数据泄露演示：在“标签完全随机”的数据上，先在全部数据上选特征再交叉验证，会得到虚高的成绩。
② 用 ColumnTransformer + Pipeline 把编码、标准化和模型串成一条流水线。
③ 网格搜索、随机搜索、Optuna 三种方法调参，比较交叉验证 AUC、测试 AUC 和用时。
④ 解释模型：置换重要性和 SHAP 值；保存最终模型。
运行：python lab9_2_pipeline_tuning.py（在 labs/ch09 目录下），约 1～2 分钟
"""
import logging
import time
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ① 数据泄露：1000 个随机特征、标签随机，真实的准确率只能是 50% 左右
rng = np.random.default_rng(0)
Xr, yr = rng.normal(size=(200, 1000)), rng.integers(0, 2, 200)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
Xsel = SelectKBest(f_classif, k=20).fit_transform(Xr, yr)              # 错误：用了全部数据（含验证折）的标签
wrong = cross_val_score(LogisticRegression(max_iter=1000), Xsel, yr, cv=skf).mean()
right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)),
                        Xr, yr, cv=skf).mean()
print("① 数据泄露演示（200 条样本、1000 个随机特征、标签随机）")
print(f"  先在全部数据上选 20 个特征，再交叉验证：准确率 {wrong:.3f}  ← 虚高")
print(f"  把选特征放进流水线，每一折只用训练部分选：准确率 {right:.3f}  ← 真实水平")

# ② 流水线
df = pd.read_csv("../data/telco_churn.csv")
y = (df["是否流失"] == "是").astype(int)
X = df.drop(columns=["客户编号", "是否流失"])
num = ["年龄", "在网月数", "月消费", "累计消费", "近半年投诉次数"]
cat = ["合约类型", "宽带类型", "技术支持", "视频会员", "付款方式"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

prep = ColumnTransformer([("num", StandardScaler(), num),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
lr = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])
cv_auc = cross_val_score(lr, X_train, y_train, cv=skf, scoring="roc_auc").mean()
lr.fit(X_train, y_train)
print(f"\n② 流水线：数值列 {len(num)} 个标准化，类别列 {len(cat)} 个独热编码 → "
      f"{len(lr.named_steps['prep'].get_feature_names_out())} 个特征 → 逻辑回归")
print(f"  逻辑回归：交叉验证 AUC {cv_auc:.3f}，测试 AUC {roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]):.3f}")
one = X_test.iloc[[0]]
print(f"  直接把原始表格的一行交给流水线：{one.iloc[0].to_dict()} → 流失概率 {lr.predict_proba(one)[0, 1]:.3f}")

# ③ 调参：梯度提升树
prep_tree = ColumnTransformer([("num", "passthrough", num),
                               ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
gb = Pipeline([("prep", prep_tree), ("model", HistGradientBoostingClassifier(random_state=42))])
base = cross_val_score(gb, X_train, y_train, cv=skf, scoring="roc_auc").mean()
print(f"\n③ 梯度提升树调参（5 折交叉验证 AUC；默认参数 {base:.3f}）")
print(f"  {'方法':<10}{'试验次数':>6}{'交叉验证 AUC':>12}{'测试 AUC':>10}{'用时（秒）':>10}   最好的参数")
results = {}

grid = {"model__learning_rate": [0.03, 0.1, 0.3], "model__max_depth": [2, 3, 5, None],
        "model__min_samples_leaf": [20, 50, 100]}
t = time.perf_counter()
gs = GridSearchCV(gb, grid, cv=skf, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)
results["网格搜索"] = (len(gs.cv_results_["params"]), gs.best_score_, gs, time.perf_counter() - t, gs.best_params_)

dist = {"model__learning_rate": loguniform(0.01, 0.5), "model__max_depth": randint(2, 8),
        "model__min_samples_leaf": randint(10, 200), "model__l2_regularization": loguniform(1e-3, 10)}
t = time.perf_counter()
rs = RandomizedSearchCV(gb, dist, n_iter=36, cv=skf, scoring="roc_auc", n_jobs=-1, random_state=42).fit(X_train, y_train)
results["随机搜索"] = (36, rs.best_score_, rs, time.perf_counter() - t, rs.best_params_)


def objective(trial):
    gb.set_params(model__learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
                  model__max_depth=trial.suggest_int("max_depth", 2, 8),
                  model__min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200),
                  model__l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True))
    return cross_val_score(gb, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1).mean()


t = time.perf_counter()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=36)
best_gb = gb.set_params(**{f"model__{k}": v for k, v in study.best_params.items()}).fit(X_train, y_train)
results["Optuna"] = (36, study.best_value, best_gb, time.perf_counter() - t, study.best_params)

for name, (n, score, model, sec, params) in results.items():
    test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    p = {k.replace("model__", ""): (round(float(v), 3) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, np.integer) else v) for k, v in params.items()}
    print(f"  {name:<10}{n:>6}{score:>12.3f}{test:>10.3f}{sec:>10.1f}   {p}")
trials = [tr.value for tr in study.trials]
print("  Optuna 前 12 次中最好 {:.3f}，前 24 次 {:.3f}，36 次 {:.3f}".format(max(trials[:12]), max(trials[:24]), max(trials)))

# ④ 解释与保存
final = results["Optuna"][2]
pi = permutation_importance(final, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
print("\n④ 置换重要性（把一列打乱后测试 AUC 下降多少）：")
print("  " + "，".join(f"{k} {v:.3f}" for k, v in imp.items()))

Z_test = final.named_steps["prep"].transform(X_test)
names = [n.split("__")[1] for n in final.named_steps["prep"].get_feature_names_out()]
explainer = shap.TreeExplainer(final.named_steps["model"])
sv = explainer.shap_values(Z_test)
base_value = float(np.ravel(explainer.expected_value)[0])
prob = final.predict_proba(X_test)[:, 1]
i = int(np.argmax(prob))
row = pd.Series(sv[i], index=names).sort_values(key=abs, ascending=False)
print(f"\n  SHAP：测试集中流失概率最高的客户（第 {i} 条，概率 {prob[i]:.3f}）")
print("  " + "，".join(f"{k}={v}" for k, v in X_test.iloc[i].items()))
print(f"  基准值（平均预测，对数几率）{base_value:.3f}；贡献最大的 5 项：")
for k, v in row.head(5).items():
    print(f"    {k:<14}{v:+.3f}")
print(f"  基准值 + 全部贡献 = {base_value + sv[i].sum():.3f}（对数几率）→ 概率 {1 / (1 + np.exp(-(base_value + sv[i].sum()))):.3f}")

shap.summary_plot(sv, Z_test, feature_names=names, show=False, max_display=10, plot_size=(7, 4.2))
plt.tight_layout(); plt.savefig("lab9_2_shap.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(range(1, 37), np.maximum.accumulate(trials), "o-", ms=3, label="Optuna 至今最好")
ax.scatter(range(1, 37), trials, s=10, c="gray", label="每次试验")
ax.axhline(base, color="C3", ls="--", lw=1, label="默认参数")
ax.set_xlabel("试验次数"), ax.set_ylabel("交叉验证 AUC"), ax.legend(fontsize=8), ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("lab9_2_optuna.png", dpi=120)

joblib.dump(final, "churn_model.joblib")
print(f"\n  模型已保存到 churn_model.joblib（{__import__('os').path.getsize('churn_model.joblib') / 1024:.0f} KB），"
      f"包含预处理和模型两步")
