"""实验 1-3 写规则 vs 学规则

同一份贷款审批数据：先手写 if-else 审批规则，再让决策树从数据里学规则，比较两者在新数据上的准确率。
运行：python lab1_3_rules_vs_learning.py（在 labs/ch01 目录下）
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("../data/loan_apply.csv")
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["是否批准"])
print(f"历史记录 {len(train)} 条（用来写规则或训练），新申请 {len(test)} 条（用来检验）")
print(f"历史记录中批准率 {(train['是否批准'] == '是').mean():.1%}")


def rule_v1(row):
    """第一版：师傅口诀，有房就批，没房看收入"""
    if row["有房产"] == "是":
        return "是"
    return "是" if row["月收入"] >= 8000 else "否"


def rule_v2(row):
    """第二版：看了几十条被拒的记录后加上信用和负债"""
    if row["信用等级"] == "一般" and row["近两年逾期次数"] >= 2:
        return "否"
    if row["负债率"] > 0.5:
        return "否"
    if row["有房产"] == "是":
        return "是"
    return "是" if row["月收入"] >= 6000 or row["信用等级"] == "非常好" else "否"


def accuracy(rule, data):
    return (data.apply(rule, axis=1) == data["是否批准"]).mean()


print("\n方法                      历史记录准确率  新申请准确率")
for name, rule in [("手写规则 第一版（2 条）", rule_v1), ("手写规则 第二版（5 条）", rule_v2)]:
    print(f"{name:<20}{accuracy(rule, train):>10.3f}{accuracy(rule, test):>12.3f}")

# 让计算机从历史记录里学规则
def encode(data):
    X = data.drop(columns="是否批准").copy()
    X["有房产"] = (X["有房产"] == "是").astype(int)
    X["信用等级"] = X["信用等级"].map({"一般": 0, "好": 1, "非常好": 2})
    return X


tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)
tree.fit(encode(train), train["是否批准"])
print(f"{'决策树学到的规则（深度 6）':<17}{tree.score(encode(train), train['是否批准']):>10.3f}"
      f"{tree.score(encode(test), test['是否批准']):>12.3f}")
print(f"\n决策树一共学出 {tree.get_n_leaves()} 条规则，前几条：")
print("\n".join(export_text(tree, feature_names=list(encode(train).columns)).splitlines()[:14]))
