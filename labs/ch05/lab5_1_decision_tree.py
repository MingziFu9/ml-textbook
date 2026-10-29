"""实验 5-1 手算信息增益并训练决策树

第一部分：用 15 条贷款申请数据复算信息熵、信息增益、信息增益率和基尼系数。
第二部分：用 2000 条模拟贷款审批数据训练 scikit-learn 决策树，观察树的深度与过拟合。
运行：python lab5_1_decision_tree.py（在 labs/ch05 目录下）
"""
from collections import Counter
from math import log2

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 第一部分：手算复核 ----------
small = pd.read_csv("../data/loan_small.csv")
target = "是否批准"


def entropy(labels):
    """信息熵：-Σ p·log2(p)"""
    n = len(labels)
    return -sum(c / n * log2(c / n) for c in Counter(labels).values())


def gini(labels):
    """基尼系数：1 - Σ p²"""
    n = len(labels)
    return 1 - sum((c / n) ** 2 for c in Counter(labels).values())


h_d = entropy(small[target])
print(f"数据集的信息熵 H(D) = {h_d:.3f}，基尼系数 = {gini(small[target]):.3f}")
print(f"{'特征':<8}{'信息增益':>8}{'增益率':>8}{'划分后基尼':>10}")
for feature in ["年龄", "有工作", "有自己的房子", "信贷情况"]:
    groups = small.groupby(feature)[target]
    weights = groups.size() / len(small)
    cond_h = sum(weights[v] * entropy(g) for v, g in groups)
    cond_gini = sum(weights[v] * gini(g) for v, g in groups)
    iv = -sum(w * log2(w) for w in weights)
    gain = h_d - cond_h
    print(f"{feature:<8}{gain:>10.3f}{gain / iv:>10.3f}{cond_gini:>12.3f}")

# ---------- 第二部分：scikit-learn 决策树 ----------
df = pd.read_csv("../data/loan_apply.csv")
df["有房产"] = (df["有房产"] == "是").astype(int)  # 两个取值的特征直接变成 0/1
X = pd.get_dummies(df.drop(columns=target), columns=["信用等级"], dtype=int)  # 三个取值的做独热编码
y = (df[target] == "是").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\n训练集 {len(X_train)} 条，测试集 {len(X_test)} 条，特征 {X.shape[1]} 个")

print(f"{'最大深度':<6}{'叶子数':>6}{'训练集准确率':>10}{'测试集准确率':>10}")
records = []
for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    tr, te = tree.score(X_train, y_train), tree.score(X_test, y_test)
    records.append((tree.get_depth(), tr, te))
    label = "不限" if depth is None else depth
    print(f"{label!s:<8}{tree.get_n_leaves():>6}{tr:>12.3f}{te:>12.3f}")

# 画出深度与准确率的关系
depths, train_acc, test_acc = zip(*records)
plt.figure(figsize=(6, 4))
plt.plot(depths, train_acc, "o-", label="训练集")
plt.plot(depths, test_acc, "s-", label="测试集")
plt.xlabel("树的实际深度")
plt.ylabel("准确率")
plt.title("决策树深度与过拟合")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lab5_1_depth.png", dpi=150)

# 训练一棵深度为 3 的树，画出来并打印规则
tree3 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42).fit(X_train, y_train)
print(f"\n深度 3 的树：测试集准确率 {tree3.score(X_test, y_test):.3f}")
print(export_text(tree3, feature_names=list(X.columns), class_names=["不批准", "批准"]))
plt.figure(figsize=(14, 6))
plot_tree(tree3, feature_names=list(X.columns), class_names=["不批准", "批准"],
          filled=True, rounded=True, fontsize=8, impurity=True)
plt.tight_layout()
plt.savefig("lab5_1_tree.png", dpi=150)
print("图片已保存：lab5_1_depth.png、lab5_1_tree.png")
