"""实验 7-2 朴素贝叶斯垃圾短信过滤

UCI SMS Spam Collection（5574 条英文短信，垃圾短信 747 条）：用词袋表示短信，训练多项式朴素贝叶斯，
查看最能代表垃圾短信的词，比较拉普拉斯平滑的作用，与逻辑回归对比，分析被误判的短信。
运行：python lab7_2_naive_bayes_sms.py（在 labs/ch07 目录下）
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

pd.set_option("display.width", 200)
df = pd.read_csv("../data/sms_spam.csv")
y = (df["标签"] == "spam").astype(int)
print(f"共 {len(df)} 条短信，垃圾短信 {y.sum()} 条（{y.mean():.1%}）")
X_train_txt, X_test_txt, y_train, y_test = train_test_split(df["短信"], y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集 {len(y_train)} 条，测试集 {len(y_test)} 条（垃圾短信 {y_test.sum()} 条）")

# ① 词袋：每条短信变成“每个词出现了几次”的向量
vec = CountVectorizer()
X_train = vec.fit_transform(X_train_txt)          # 只用训练集建词表
X_test = vec.transform(X_test_txt)
print(f"\n① 词袋：训练集词表 {len(vec.vocabulary_)} 个词；矩阵 {X_train.shape[0]} × {X_train.shape[1]}，"
      f"非零元素只占 {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.2%}")
ex = X_train_txt.iloc[0]
print(f"  例：“{ex}”")
print("  →", {w: int(c) for w, c in zip(vec.get_feature_names_out()[X_train[0].indices], X_train[0].data)})


def report(name, model, Xtr, Xte):
    t = time.perf_counter()
    model.fit(Xtr, y_train)
    sec = time.perf_counter() - t
    pred = model.predict(Xte)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"  {name:<22}准确率 {(tp + tn) / len(pred):.3f}  精确率 {precision_score(y_test, pred):.3f}  "
          f"召回率 {recall_score(y_test, pred):.3f}  漏判 {fn:>3}  误判 {fp:>2}  用时 {sec:.3f} 秒")
    return pred


print("\n② 模型对比（测试集）")
nb = MultinomialNB()                                # alpha=1.0，即拉普拉斯平滑
pred_nb = report("多项式朴素贝叶斯", nb, X_train, X_test)
report("逻辑回归", LogisticRegression(max_iter=2000), X_train, X_test)

# ③ 每个词在两类中的条件概率
words = vec.get_feature_names_out()
logp = nb.feature_log_prob_                         # 第 0 行：P(词 | 正常)，第 1 行：P(词 | 垃圾)
ratio = logp[1] - logp[0]
print(f"\n③ 先验概率：P(垃圾) = {np.exp(nb.class_log_prior_[1]):.3f}，P(正常) = {np.exp(nb.class_log_prior_[0]):.3f}")
print("  最能代表垃圾短信的词（P(词|垃圾) ÷ P(词|正常) 最大，且在训练集中出现至少 20 次）：")
cnt = np.asarray(X_train.sum(axis=0)).ravel()
order = [i for i in np.argsort(-ratio) if cnt[i] >= 20][:10]
print("   " + "，".join(f"{words[i]}（{np.exp(ratio[i]):.0f} 倍）" for i in order))
order = [i for i in np.argsort(ratio) if cnt[i] >= 20][:10]
print("  最能代表正常短信的词：")
print("   " + "，".join(f"{words[i]}（{np.exp(-ratio[i]):.0f} 倍）" for i in order))
for w in ["free", "call", "you", "love"]:
    i = vec.vocabulary_[w]
    print(f"  P({w} | 垃圾) = {np.exp(logp[1, i]):.5f}，P({w} | 正常) = {np.exp(logp[0, i]):.5f}")

# ④ 拉普拉斯平滑
print("\n④ 拉普拉斯平滑的作用")
spam_cnt = np.asarray(X_train[y_train.to_numpy() == 1].sum(axis=0)).ravel()
zero_in_spam = (spam_cnt == 0).sum()
print(f"  词表中有 {zero_in_spam} 个词从没在训练集的垃圾短信里出现过（占 {zero_in_spam / len(words):.0%}）")
for a in [1e-10, 0.01, 0.1, 1.0]:
    m = MultinomialNB(alpha=a).fit(X_train, y_train)
    pred = m.predict(X_test)
    print(f"  alpha = {a:<6g} 准确率 {(pred == y_test).mean():.3f}  精确率 {precision_score(y_test, pred):.3f}  "
          f"召回率 {recall_score(y_test, pred):.3f}")
m0 = MultinomialNB(alpha=1e-10).fit(X_train, y_train)
msg = next(t for t in X_test_txt if t.startswith("Dear Voucher Holder"))
x = vec.transform([msg])
zero = [words[i] for i in x.indices if spam_cnt[i] == 0]
print(f"  例：“{msg[:80]}……”")
print(f"  其中 {zero} 从没在训练集的垃圾短信里出现过；alpha ≈ 0 时 P(这些词 | 垃圾) ≈ 0，"
      f"整条短信判为垃圾的概率 {m0.predict_proba(x)[0, 1]:.3f}；alpha = 1 时为 {nb.predict_proba(x)[0, 1]:.3f}")

# ⑤ 被误判的短信
prob = nb.predict_proba(X_test)[:, 1]
res = pd.DataFrame({"短信": X_test_txt.to_numpy(), "真实": y_test.to_numpy(), "垃圾概率": prob.round(3)})
fn = res[(res["真实"] == 1) & (pred_nb == 0)].sort_values("垃圾概率")
fp = res[(res["真实"] == 0) & (pred_nb == 1)].sort_values("垃圾概率", ascending=False)
# 漏判的短信中约一半是用聊天口吻写的交友类付费短信，内容不宜印在教材里，这里只列出几条有代表性的
picked = fn[fn["短信"].str.contains("missed your call|40 years|dartboard|Xmas|perfect partner")]
print(f"\n⑤ 漏判的垃圾短信共 {len(fn)} 条，其中 5 条：")
for _, r in picked.iterrows():
    print(f"  [{r['垃圾概率']:.3f}] {r['短信'][:90]}")
print(f"  误判为垃圾的正常短信共 {len(fp)} 条：")
for _, r in fp.iterrows():
    print(f"  [{r['垃圾概率']:.3f}] {r['短信'][:90]}")
