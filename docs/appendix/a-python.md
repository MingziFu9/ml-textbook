---
title: 附录 A Python 速查
description: 本书实验中常用的 Python 基础语法、NumPy、pandas、Matplotlib 和 scikit-learn 通用用法速查，以及常见报错的原因和解决办法
---

# 附录 A Python 速查

本附录汇总了本书实验中最常用的 Python 知识点，供做实验时随时查阅。它不是完整的 Python 教程，如果你还没有学过 Python，建议先学习“Python 程序设计”课程。代码后面的注释 `# ->` 表示运行结果，都已在 Python 3.11 下运行核对。

## A.1 基础语法

```python
scores = [85, 92, 78, 92]
scores.append(60)                    # 末尾添加
scores[0], scores[-1]                # 按位置取：第一个、最后一个 -> (85, 60)
scores[1:3]                          # 切片，含头不含尾 -> [92, 78]
len(scores), sum(scores), max(scores)   # -> (5, 407, 92)
[s for s in scores if s >= 80]       # 列表推导式：筛选 -> [85, 92, 92]
sorted(scores, reverse=True)         # 从大到小排序 -> [92, 92, 85, 78, 60]

price = {"苹果": 6.5, "香蕉": 3.0}   # 字典：键 -> 值
price["梨"] = 4.2                    # 添加或修改
price.get("西瓜", 0)                 # 取值，没有这个键时返回默认值 -> 0
for name, p in price.items():        # 同时取键和值
    print(f"{name}：{p:.2f} 元")      # f-string，保留两位小数 -> 苹果：6.50 元 ……

def mae(y_true, y_pred):             # 定义函数：平均绝对误差
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

mae([2000, 2500, 3000], [2100, 2400, 2700])   # -> 166.66666666666666
```

`zip` 把两个列表按位置配对；`enumerate(列表)` 同时给出序号和元素；`range(0, 10, 2)` 生成 0、2、4、6、8。

## A.2 NumPy

NumPy 的数组（`ndarray`）是机器学习中表示数据的基本格式：一行一个样本，一列一个特征。

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape, a.ndim, a.dtype             # 形状、维数、元素类型 -> ((2, 3), 2, dtype('int64'))
a[0, 1], a[:, 0], a[1]               # 取元素、取第 0 列、取第 1 行 -> 2, [1 4], [4 5 6]
a.reshape(3, 2).shape                # 改变形状 -> (3, 2)
a.T.shape                            # 转置 -> (3, 2)
a.sum(), a.sum(axis=0), a.mean(axis=1)   # 全部求和、按列求和、按行求平均 -> 21, [5 7 9], [2. 5.]
a * 2, a + np.array([10, 20, 30])    # 逐元素运算；形状不同时自动“广播” -> [[11 22 33] [14 25 36]]
a[a > 3]                             # 布尔索引：取出大于 3 的元素 -> [4 5 6]
a @ np.array([1, 0, 1])              # 矩阵乘法（每行与权重做加权求和）-> [ 4 10]

np.zeros((2, 3)), np.ones(3), np.arange(0, 1, 0.25)   # 全 0、全 1、等差数列 -> [0. 0.25 0.5 0.75]
np.linspace(0, 1, 5)                 # 0 到 1 均匀取 5 个点 -> [0. 0.25 0.5 0.75 1.]
rng = np.random.default_rng(42)      # 固定随机种子，结果可以重现
rng.normal(0, 1, size=3).round(3)    # 3 个标准正态随机数 -> [ 0.305 -1.04   0.75 ]
np.argmax([0.1, 0.7, 0.2])           # 最大值所在的位置（分类模型取预测类别）-> 1
```

`axis=0` 是沿着行的方向（对每一列算），`axis=1` 是沿着列的方向（对每一行算）。

## A.3 pandas

pandas 的 `DataFrame` 是带列名的表格，读数据、清洗、统计都用它。

```python
import pandas as pd

df = pd.DataFrame({"区域": ["天河", "海珠", "天河", "白云"],
                   "面积": [89.0, 65.5, None, 102.0],
                   "总价": [520, 310, 450, 380]})
df.shape                             # -> (4, 3)
df.head(2)                           # 前 2 行；df.info() 看各列类型和非空个数；df.describe() 看统计量
df["面积"]                           # 取一列（Series）
df[["区域", "总价"]]                 # 取几列（DataFrame）
df[df["总价"] > 400]                 # 按条件筛选行
df.loc[0, "总价"], df.iloc[0, 2]     # 按标签、按位置取值 -> 520, 520
df.isna().sum()                      # 每列缺失值个数：面积 1
df["面积"] = df["面积"].fillna(df["面积"].median())   # 用中位数填充 -> 89.0
df["单价"] = (df["总价"] * 10000 / df["面积"]).round()  # 新增一列
df.groupby("区域")["总价"].mean()    # 分组统计：天河 485.0，海珠 310.0，白云 380.0
df["区域"].value_counts()            # 每个取值出现几次：天河 2
df.sort_values("总价", ascending=False)   # 按总价从高到低排序
pd.get_dummies(df["区域"])           # 独热编码，每个区域一列
df.to_csv("out.csv", index=False)    # 保存，不写行号
```

读数据：`pd.read_csv("../data/telco_churn.csv")`；中文 CSV 在 Excel 里乱码时，保存用 `encoding="utf-8-sig"`。

**日期与时间序列**（第 12 章）：

```python
s = pd.Series([10, 12, 11, 13, 15], index=pd.date_range("2025-12-01", periods=5, freq="D"))
s.index.day_of_week                  # 星期几，0 是周一 -> [0 1 2 3 4]
s.shift(1).tolist()                  # 往后错开 1 天（滞后特征）-> [nan, 10.0, 12.0, 11.0, 13.0]
s.rolling(3).mean().tolist()         # 3 天滑动平均 -> [nan, nan, 11.0, 12.0, 13.0]
s.diff().tolist()                    # 一阶差分 -> [nan, 2.0, -1.0, 2.0, 2.0]
s.resample("W").sum()                # 按周汇总
```

读入时用 `pd.read_csv(文件, parse_dates=["日期"], index_col="日期")` 把日期列变成日期索引。

## A.4 Matplotlib

```python
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False        # 正常显示负号

fig, ax = plt.subplots(1, 2, figsize=(10, 4))     # 1 行 2 列两张子图
ax[0].plot([1, 2, 3], [0.9, 0.6, 0.4], marker="o", label="训练损失")
ax[0].set_xlabel("轮次"); ax[0].set_ylabel("损失"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].scatter([1, 2, 3, 4], [3, 5, 4, 6], c=[0, 1, 0, 1], cmap="coolwarm")   # 散点图，按类别着色
plt.tight_layout()
plt.savefig("result.png", dpi=100)                # 保存图片；在 Jupyter 中直接 plt.show()
```

常用图：`hist` 直方图（看分布）、`boxplot` 箱线图（看异常值）、`bar` / `barh` 柱状图（比较类别、特征重要性）、`imshow` 热力图和图片（混淆矩阵、相关系数矩阵、卷积特征图）。

## A.5 scikit-learn 的通用用法

scikit-learn 中所有模型的用法都一样，记住“三件套”：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_tr)          # 预处理器：只在训练集上 fit
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

model = LogisticRegression(max_iter=1000)    # 1. 创建模型，设超参数
model.fit(X_tr_s, y_tr)                      # 2. 训练
model.predict(X_te_s[:3])                    # 3. 预测类别 -> [2 1 1]
model.predict_proba(X_te_s[:1]).round(3)     # 每一类的概率
model.score(X_te_s, y_te)                    # 测试集准确率 -> 0.9111111111111111
```

- 预处理器（缩放、编码）用 `fit` 学参数、`transform` 变换，**只在训练集上 fit**，否则会数据泄露（第 2 章）；
- 换一个模型只要换第 1 行，比如 `RandomForestClassifier()`、`SVC()`；
- 评价指标在 `sklearn.metrics` 中，如 `accuracy_score`、`classification_report`、`mean_absolute_error`；
- 把预处理和模型串起来用 `Pipeline`（第 9 章），交叉验证用 `cross_val_score`。

## A.6 常见错误速查

| 报错或现象 | 常见原因 | 解决办法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'xxx'` | 没装这个库，或者装在了另一个 Python 环境 | 在当前环境运行 `pip install xxx`；Jupyter 中可运行 `%pip install xxx`。注意 `sklearn` 的安装名是 `scikit-learn` |
| `FileNotFoundError` | 当前目录不对，相对路径找不到文件 | 在实验代码所在目录运行，或用 `print(os.getcwd())` 查看当前目录 |
| `KeyError: '列名'` | 列名拼错，或含有空格、全角符号 | `print(df.columns.tolist())` 查看真实列名 |
| `ValueError: could not convert string to float` | 特征中还有文字，没有编码 | 用独热编码或标签编码把类别型特征转成数字 |
| `ValueError: Input X contains NaN` | 有缺失值，模型不接受 | 先 `df.isna().sum()` 找出来，再填充或删除 |
| `ValueError: Expected 2D array, got 1D array` | 只有一个特征时传了一维数组 | `X.reshape(-1, 1)`，或用 `df[["面积"]]`（两层方括号） |
| `ValueError: shapes ... not aligned` 或 `mat1 and mat2 shapes cannot be multiplied` | 矩阵乘法或全连接层的形状对不上 | 打印 `.shape` 逐步检查，核对每层的输入大小 |
| `ConvergenceWarning` | 迭代次数不够，没有收敛 | 先做特征缩放，再加大 `max_iter` |
| 图中中文显示成方框 | 没有中文字体 | 按 A.4 设置 `font.sans-serif`，Linux 安装 `fonts-noto-cjk` |
| 每次运行结果不一样 | 有随机划分或随机初始化 | 设置 `random_state=42`，PyTorch 用 `torch.manual_seed(0)` |
| `SettingWithCopyWarning` | 在筛选出的子表上直接改值 | 先 `.copy()`，或用 `df.loc[条件, "列"] = 值` |
