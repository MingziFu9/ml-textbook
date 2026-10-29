"""实验 1-2 第一个机器学习模型：鸢尾花分类

加载数据 → 划分训练集和测试集 → 训练 → 预测 → 评估，走一遍完整流程。
运行：python lab1_2_iris.py（在 labs/ch01 目录下）
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 第 1 步：加载数据，转成 pandas 表格看一看
iris = load_iris()
names = {0: "山鸢尾", 1: "变色鸢尾", 2: "维吉尼亚鸢尾"}
df = pd.DataFrame(iris.data, columns=["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"])
df["品种"] = [names[i] for i in iris.target]
print(f"共 {len(df)} 朵花，{df.shape[1] - 1} 个特征（单位：厘米）")
print(df.sample(5, random_state=1).to_string())
print("\n每个品种的数量：", df["品种"].value_counts().to_dict())
print("\n各品种花瓣长度的平均值：", df.groupby("品种")["花瓣长度"].mean().round(2).to_dict())

# 第 2 步：划分训练集（80%）和测试集（20%）
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n训练集 {len(X_train)} 条，测试集 {len(X_test)} 条")

# 第 3 步：训练。K 近邻：找训练集里最像的 5 朵花，看它们多数是什么品种
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 第 4 步：预测并评估
pred = model.predict(X_test)
print(f"测试集准确率：{accuracy_score(y_test, pred):.3f}（{(pred == y_test).sum()}/{len(y_test)}）")
print("混淆矩阵（行是真实品种，列是预测品种）：")
print(pd.DataFrame(confusion_matrix(y_test, pred), index=list(names.values()), columns=list(names.values())).to_string())

# 第 5 步：用模型判断一朵新花
new_flower = [[5.0, 3.4, 1.5, 0.2], [6.3, 2.8, 5.1, 1.6]]
for f, p, prob in zip(new_flower, model.predict(new_flower), model.predict_proba(new_flower)):
    print(f"新花 {f} → {names[p]}（5 个邻居中 {int(prob.max() * 5)} 个是这个品种）")
