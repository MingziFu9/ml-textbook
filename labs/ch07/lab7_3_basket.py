"""实验 7-3 超市购物篮分析

3000 张社区超市购物小票：用 mlxtend 的 Apriori 找频繁项集、生成关联规则，按提升度筛选，
识别“置信度高但没有意义”的规则，与 FP-Growth 比较，最后做一个简单的“买了又买”推荐。
运行：python lab7_3_basket.py（在 labs/ch07 目录下）。需要先安装 mlxtend：pip install mlxtend
"""
import time

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

pd.set_option("display.width", 200)
df = pd.read_csv("../data/basket.csv")
transactions = df["商品"].str.split("|").tolist()
print(f"共 {len(transactions)} 张小票，平均每张 {sum(map(len, transactions)) / len(transactions):.1f} 件商品")
print("前 3 张：", transactions[:3])

# ① 转成 0/1 表：每行一张小票，每列一种商品
te = TransactionEncoder()
X = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
print(f"\n① 0/1 表：{X.shape[0]} 行 × {X.shape[1]} 列（商品种数）")
top = X.mean().sort_values(ascending=False)
print("  单个商品的支持度（前 8）：" + "，".join(f"{k} {v:.3f}" for k, v in top.head(8).items()))

# ② 手算一条规则，和程序结果对照
a, b = "面包", "牛奶"
n_a, n_b, n_ab = X[a].sum(), X[b].sum(), (X[a] & X[b]).sum()
print(f"\n② 规则 {{{a}}} → {{{b}}}：含{a} {n_a} 张，含{b} {n_b} 张，同时含两者 {n_ab} 张")
print(f"  支持度 = {n_ab}/{len(X)} = {n_ab / len(X):.3f}；置信度 = {n_ab}/{n_a} = {n_ab / n_a:.3f}；"
      f"提升度 = {n_ab / n_a:.3f} ÷ {n_b / len(X):.3f} = {n_ab / n_a / (n_b / len(X)):.2f}")

# ③ Apriori：最小支持度 0.02
t = time.perf_counter()
freq = apriori(X, min_support=0.02, use_colnames=True)
t_ap = time.perf_counter() - t
freq["长度"] = freq["itemsets"].apply(len)
print(f"\n③ Apriori（最小支持度 0.02）：频繁项集 {len(freq)} 个，其中 1 项 {(freq['长度'] == 1).sum()} 个、"
      f"2 项 {(freq['长度'] == 2).sum()} 个、3 项 {(freq['长度'] == 3).sum()} 个、4 项 {(freq['长度'] == 4).sum()} 个")
show = lambda s: "、".join(sorted(s))
print("  支持度最高的 3 项及以上项集：")
for _, r in freq[freq["长度"] >= 3].sort_values("support", ascending=False).head(3).iterrows():
    print(f"    {{{show(r['itemsets'])}}}  {r['support']:.3f}")

# ④ 生成规则：置信度至少 0.3，按提升度排序
rules = association_rules(freq, metric="confidence", min_threshold=0.3)
rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)].copy()
rules["规则"] = rules["antecedents"].apply(show) + " → " + rules["consequents"].apply(show)
cols = ["规则", "support", "confidence", "lift"]
print(f"\n④ 置信度 ≥ 0.3 的单商品规则共 {len(rules)} 条，按提升度排序：")
print(rules.sort_values("lift", ascending=False)[cols].round(3).to_string(index=False))

# ⑤ 置信度高、提升度低的规则：只是因为后件本身就热门
print("\n⑤ 换一个角度：后件是“鸡蛋”“矿泉水”这类热门商品的规则（置信度看上去不低，提升度却接近 1）")
r2 = association_rules(freq, metric="confidence", min_threshold=0.15)
r2 = r2[(r2["antecedents"].apply(len) == 1) & (r2["consequents"].apply(len) == 1)].copy()
r2["规则"] = r2["antecedents"].apply(show) + " → " + r2["consequents"].apply(show)
hot = r2[r2["consequents"].apply(lambda s: next(iter(s)) in {"鸡蛋", "矿泉水", "纸巾"})]
print(hot.sort_values("confidence", ascending=False)[cols].round(3).head(6).to_string(index=False))

# “纸尿裤和啤酒”
a, b = "纸尿裤", "啤酒"
s_ab = (X[a] & X[b]).mean()
print(f"\n  传说中的 {{{a}}} → {{{b}}}：支持度 {s_ab:.3f}（低于 0.02，没进频繁项集），置信度 {s_ab / X[a].mean():.3f}，"
      f"提升度 {s_ab / X[a].mean() / X[b].mean():.2f}")

# ⑥ FP-Growth：结果相同，速度不同
t = time.perf_counter()
freq_fp = fpgrowth(X, min_support=0.02, use_colnames=True)
t_fp = time.perf_counter() - t
same = set(map(frozenset, freq["itemsets"])) == set(map(frozenset, freq_fp["itemsets"]))
print(f"\n⑥ FP-Growth：频繁项集 {len(freq_fp)} 个，与 Apriori 完全相同：{same}")
for ms in [0.02, 0.005, 0.002]:
    t = time.perf_counter(); n1 = len(apriori(X, min_support=ms)); t1 = time.perf_counter() - t
    t = time.perf_counter(); n2 = len(fpgrowth(X, min_support=ms)); t2 = time.perf_counter() - t
    print(f"  最小支持度 {ms:<6} 频繁项集 {n1:>4} 个；Apriori {t1:.3f} 秒，FP-Growth {t2:.3f} 秒")

# ⑦ 买了又买：顾客把某件商品放进购物车时，推荐提升度最高的 3 件
print("\n⑦ “买了又买”推荐（规则前件为该商品，置信度 ≥ 0.15，按提升度取前 3）")
for item in ["火锅底料", "啤酒", "面包", "纸尿裤"]:
    rec = r2[r2["antecedents"] == frozenset([item])].sort_values("lift", ascending=False).head(3)
    print(f"  放入“{item}” → " + "，".join(f"{show(c)}（置信度 {cf:.2f}，提升度 {lf:.1f}）"
                                           for c, cf, lf in zip(rec["consequents"], rec["confidence"], rec["lift"])))
