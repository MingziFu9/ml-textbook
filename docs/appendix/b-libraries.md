---
title: 附录 B 工具库速查
description: 本书实验用到的工具库的版本、安装方法、常用接口和书中用到的位置
---

# 附录 B 工具库速查

按本书用到的工具库分节，每节给出安装命令、最常用的接口和书中用到的位置，只收录实验真正用到的接口。版本号是编者运行实验时的版本（Python 3.11），全部写在 `labs/requirements.txt` 中，一条命令即可装好：

```bash
pip install -r labs/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

| 工具库 | 版本 | 用途 | 用到的章 |
|---|---|---|---|
| NumPy | 2.4.6 | 数组与数值计算 | 全书 |
| pandas | 3.1.0 | 表格数据读取、清洗、统计，日期与时间序列 | 全书 |
| Matplotlib | 3.11.2 | 画图 | 全书 |
| scikit-learn | 1.9.1 | 经典机器学习：预处理、模型、评估、调参 | 全书 |
| XGBoost、LightGBM | 3.2.0、4.7.0 | 梯度提升树 | 第 5、12 章 |
| imbalanced-learn | 0.14.2 | SMOTE 等不平衡数据处理 | 第 4 章 |
| mlxtend | 0.25.0 | Apriori、FP-Growth 关联规则 | 第 7 章 |
| Optuna、SHAP | 5.0.0、0.51.0 | 自动调参、模型解释 | 第 9 章 |
| joblib、FastAPI、Gradio | —、0.162.0、6.31.0 | 模型保存、预测接口、网页小工具 | 第 9 章 |
| PyTorch、Keras | 2.15.0、3.15.1 | 深度学习框架 | 第 10～12 章 |
| statsmodels | 0.15.0 | 时间序列分解、平稳性检验、指数平滑、ARIMA | 第 12 章 |
| Gymnasium | 1.4.0 | 强化学习环境 | 第 12 章 |

joblib 随 scikit-learn 一起安装。

## B.1 scikit-learn

```bash
pip install scikit-learn
```

所有模型都是“创建 → `fit` → `predict`”三步（附录 A.5）。

| 接口 | 作用 | 书中用到的位置 |
|---|---|---|
| `datasets.load_iris()`、`load_breast_cancer()`、`load_wine()`、`load_digits()` | 自带的小数据集 | 实验 1-2、4-2、7-1、6-3 |
| `model_selection.train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)` | 划分训练集和测试集 | 第 1 章起 |
| `preprocessing.StandardScaler`、`MinMaxScaler` | 标准化、归一化 | 第 2 章起 |
| `preprocessing.OneHotEncoder(handle_unknown="ignore")` | 独热编码（放进流水线；第 2 章用 pandas 的 `get_dummies`） | 实验 9-2 |
| `feature_selection.VarianceThreshold`、`RFE` | 过滤法、包裹法特征选择（嵌入法用 Lasso 的系数） | 实验 2-3 |
| `linear_model.LinearRegression`、`Ridge`、`Lasso` | 线性回归、岭回归、Lasso | 第 3 章 |
| `linear_model.LogisticRegression` | 逻辑回归 | 第 4 章 |
| `tree.DecisionTreeClassifier`、`export_text`、`plot_tree` | 决策树及其可视化 | 实验 5-1 |
| `ensemble.RandomForestClassifier`、`GradientBoostingClassifier` | 随机森林、梯度提升树 | 实验 5-2、5-3 |
| `ensemble.IsolationForest` | 孤立森林异常检测 | 实验 8-2 |
| `svm.SVC(kernel="rbf", C=1, gamma="scale")`、`SVR` | 支持向量机 | 第 6 章 |
| `neighbors.KNeighborsClassifier` | K 近邻 | 实验 7-1 |
| `naive_bayes.MultinomialNB` | 多项式朴素贝叶斯 | 实验 7-2、11-3 |
| `feature_extraction.text.CountVectorizer` | 把文本变成词频向量 | 实验 7-2、11-3 |
| `cluster.KMeans`、`AgglomerativeClustering`、`DBSCAN` | 聚类 | 第 8 章 |
| `decomposition.PCA`、`manifold.TSNE` | 降维 | 实验 8-3 |
| `metrics.accuracy_score`、`precision_score`、`recall_score`、`f1_score`、`classification_report`、`confusion_matrix` | 分类评价 | 第 4 章起 |
| `metrics.roc_curve`、`roc_auc_score`、`precision_recall_curve` | ROC、PR 曲线 | 实验 4-2 |
| `metrics.mean_absolute_error`、`mean_squared_error`、`r2_score` | 回归评价 | 第 3 章 |
| `metrics.silhouette_score` | 轮廓系数 | 第 8 章 |
| `model_selection.cross_val_score`、`KFold`、`StratifiedKFold`、`TimeSeriesSplit` | 交叉验证 | 第 9、12 章 |
| `model_selection.learning_curve`、`validation_curve` | 学习曲线、验证曲线 | 实验 9-1 |
| `model_selection.GridSearchCV`、`RandomizedSearchCV` | 网格搜索、随机搜索 | 实验 9-2 |
| `pipeline.Pipeline`、`compose.ColumnTransformer` | 流水线、按列预处理 | 实验 9-2 |
| `inspection.permutation_importance` | 置换重要性 | 实验 9-2 |

## B.2 XGBoost 与 LightGBM

```bash
pip install xgboost lightgbm
```

| 接口 | 作用 | 书中用到的位置 |
|---|---|---|
| `xgboost.XGBClassifier(n_estimators, learning_rate, max_depth, early_stopping_rounds)` | XGBoost 分类 | 实验 5-3 |
| `lightgbm.LGBMClassifier(...)`、`LGBMRegressor(n_estimators, learning_rate, num_leaves)` | LightGBM 分类、回归 | 实验 5-3、12-2 |
| `model.fit(X, y, eval_set=[(X_val, y_val)])` | 训练时同时看验证集，用于早停 | 实验 5-3 |
| `model.feature_importances_`、`booster_.feature_importance("gain")` | 特征重要性 | 实验 5-3、12-2 |

两者的用法和 scikit-learn 相同，可以直接放进 `Pipeline` 和 `GridSearchCV`。LightGBM 的 `verbose=-1` 可以关掉训练时的提示信息。

## B.3 imbalanced-learn 与 mlxtend

| 接口 | 作用 | 书中用到的位置 |
|---|---|---|
| `imblearn.over_sampling.SMOTE(random_state=42).fit_resample(X, y)` | SMOTE 过采样，只用于训练集 | 实验 4-3 |
| `mlxtend.preprocessing.TransactionEncoder` | 把购物小票变成 0/1 表 | 实验 7-3 |
| `mlxtend.frequent_patterns.apriori(df, min_support=0.02, use_colnames=True)`、`fpgrowth` | 找频繁项集 | 实验 7-3 |
| `mlxtend.frequent_patterns.association_rules(freq, metric="confidence", min_threshold=0.5)` | 生成关联规则 | 实验 7-3 |

## B.4 Optuna、SHAP 与模型部署

| 接口 | 作用 | 书中用到的位置 |
|---|---|---|
| `optuna.create_study(direction="maximize")`、`study.optimize(objective, n_trials=50)` | 自动调参；`objective` 函数里用 `trial.suggest_int`、`suggest_float` 取超参数 | 实验 9-2 |
| `shap.TreeExplainer(model)`、`explainer.shap_values(X)`、`shap.summary_plot` | 计算 SHAP 值、画摘要图 | 实验 9-2 |
| `joblib.dump(model, "model.joblib")`、`joblib.load(...)` | 保存、加载模型 | 实验 9-2、9-3 |
| `fastapi.FastAPI()`、`@app.post("/predict")`，用 `uvicorn app_fastapi:app` 启动 | 把模型做成预测接口 | 实验 9-3 |
| `gr.Interface(fn, inputs=[gr.Number(...), gr.Radio(...)], outputs=gr.Label()).launch()` | 把模型做成网页小工具 | 实验 9-3 |

## B.5 PyTorch

```bash
pip install torch          # 只用 CPU 时，可按 PyTorch 官网说明安装体积更小的 CPU 版
```

| 接口 | 作用 | 书中用到的位置 |
|---|---|---|
| `torch.tensor(x, dtype=torch.float32)`、`.shape`、`.numpy()` | 张量与 NumPy 互转 | 第 10 章起 |
| `torch.nn.Linear`、`Conv2d`、`MaxPool2d`、`Embedding`、`RNN`、`LSTM`、`GRU`、`Dropout`、`BatchNorm1d` | 常用的层 | 第 10、11 章 |
| `torch.nn.ReLU()`、`Sigmoid()`、`Tanh()` | 激活函数 | 第 10 章 |
| `class Net(nn.Module)`：`__init__` 里定义层，`forward` 里写计算 | 自定义网络 | 第 10 章起 |
| `nn.CrossEntropyLoss()`、`BCEWithLogitsLoss()`、`MSELoss()` | 多分类、二分类、回归的损失 | 第 10～12 章 |
| `torch.optim.SGD`、`Adam(model.parameters(), lr=1e-3)` | 优化器 | 第 10 章起 |
| `loss.backward()`、`opt.step()`、`opt.zero_grad()` | 训练循环的核心三步 | 第 10 章起 |
| `model.train()`、`model.eval()`、`with torch.no_grad():` | 切换训练、评估模式，评估时不算梯度 | 第 10 章起 |
| `torch.manual_seed(0)` | 固定随机种子 | 第 10 章起 |
| `model.state_dict()`、`model.load_state_dict(...)` | 取出、恢复参数（早停时保存最好的一轮） | 实验 10-3、12-2 |

训练循环的写法见 10.5 节：取一批数据 → 前向计算损失 → 清零梯度、反向传播 → 更新参数。

## B.6 Keras

```bash
pip install keras torch
```

Keras 3 可以用 PyTorch、TensorFlow 或 JAX 作后端，本书用 PyTorch，要在导入前设置环境变量：

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
```

| 接口 | 作用 | 书中用到的位置 |
|---|---|---|
| `keras.Sequential([keras.Input(shape=(64,)), keras.layers.Dense(64, activation="relu"), ...])` | 按顺序堆叠层 | 实验 10-2 起 |
| `layers.Conv2D`、`MaxPooling2D`、`Flatten`、`Embedding`、`LSTM`、`Dropout` | 常用的层 | 第 11 章 |
| `model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])` | 设置优化器、损失和指标 | 实验 10-2 起 |
| `model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, callbacks=[...])` | 训练 | 实验 10-2 起 |
| `keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)` | 早停 | 第 10 章实操题 |
| `model.evaluate`、`model.predict`、`model.count_params()` | 评估、预测、参数个数 | 实验 10-2 起 |
| `keras.utils.set_random_seed(0)` | 固定随机种子 | 实验 10-2 起 |
| `keras.datasets.fashion_mnist.load_data()` | 下载 Fashion-MNIST | 实验 11-2 |

## B.7 statsmodels

```bash
pip install statsmodels
```

| 接口 | 作用 | 书中用到的位置 |
|---|---|---|
| `tsa.seasonal.seasonal_decompose(y, model="multiplicative", period=7)` | 时间序列分解 | 实验 12-1 |
| `tsa.stattools.adfuller(y)[1]`、`kpss(y)[1]` | 平稳性检验的 p 值 | 实验 12-1 |
| `tsa.stattools.acf(y, nlags=28)`、`graphics.tsaplots.plot_acf` | 自相关系数、自相关图 | 实验 12-1 |
| `tsa.holtwinters.SimpleExpSmoothing`、`ExponentialSmoothing(trend="add", seasonal="mul", seasonal_periods=7)` | 指数平滑、霍尔特—温特斯 | 实验 12-1、12-2 |
| `tsa.statespace.sarimax.SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s))` | ARIMA、SARIMA | 实验 12-1 |
| `.fit()`、`.forecast(28)`、`.aic`、`.params` | 拟合、预测未来若干期、AIC、学到的参数 | 实验 12-1 |

## B.8 Gymnasium

```bash
pip install gymnasium
```

| 接口 | 作用 | 书中用到的位置 |
|---|---|---|
| `gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)` | 创建冰湖环境 | 实验 12-3 |
| `env.observation_space.n`、`env.action_space.n` | 状态个数、动作个数 | 实验 12-3 |
| `s, info = env.reset(seed=0)` | 开始新回合，回到起点 | 实验 12-3 |
| `s2, r, terminated, truncated, info = env.step(a)` | 执行一个动作 | 实验 12-3 |
| `env.action_space.sample()` | 随机取一个动作 | 实验 12-3 |

Gymnasium 是旧库 `gym` 的继任者，`step` 返回 5 个值（旧版 4 个），网上的旧代码要相应修改。
