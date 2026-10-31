"""实验 11-3 LSTM 短信分类

沿用第 7 章的垃圾短信数据（5574 条英文短信）和同样的划分（测试集 30%，random_state=42）。
① 把短信变成整数序列：分词、建词表、截断与补齐；
② PyTorch：词嵌入 + 简单 RNN、词嵌入 + LSTM，两种循环网络对比；
③ 同一个 LSTM 用 Keras 3 再写一遍；
④ 与第 7 章的多项式朴素贝叶斯比较效果和速度。
为保护隐私和避免不良内容，程序不打印数据集中的短信原文，只用两条自编的短信演示预测。
运行：python lab11_3_lstm_sms.py（在 labs/ch11 目录下），约 1 分钟
"""
import os
import re
import time
from collections import Counter

os.environ["KERAS_BACKEND"] = "torch"
import keras                                     # noqa: E402
import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402
import torch                                     # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer   # noqa: E402
from sklearn.metrics import confusion_matrix     # noqa: E402
from sklearn.model_selection import train_test_split          # noqa: E402
from sklearn.naive_bayes import MultinomialNB    # noqa: E402
from torch import nn                             # noqa: E402

torch.set_num_threads(4)
df = pd.read_csv("../data/sms_spam.csv")
y = (df["标签"] == "spam").astype(int).values
txt_tr, txt_te, y_tr, y_te = train_test_split(df["短信"], y, test_size=0.3, random_state=42, stratify=y)
txt_tr, txt_val, y_tr, y_val = train_test_split(txt_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)
print(f"训练 {len(y_tr)} 条，验证 {len(y_val)} 条，测试 {len(y_te)} 条（测试集中垃圾短信 {y_te.sum()} 条）")

# ① 文本 → 整数序列
tok = lambda s: re.findall(r"[a-z0-9']+", s.lower())          # noqa: E731  小写、按非字母数字切开
cnt = Counter(w for s in txt_tr for w in tok(s))
vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common()) if c >= 2}   # 0 留给补齐，1 留给生词
MAXLEN = 40
lens = np.array([len(tok(s)) for s in txt_tr])


def encode(s):
    ids = [vocab.get(w, 1) for w in tok(s)][-MAXLEN:]           # 太长：只保留最后 40 个词
    return [0] * (MAXLEN - len(ids)) + ids                      # 太短：在前面补 0


print(f"\n① 词表：训练集出现至少 2 次的词 {len(vocab)} 个（另有补齐符 0、生词 1）")
print(f"  短信长度（词数）：中位数 {np.median(lens):.0f}，90% 的短信不超过 {np.percentile(lens, 90):.0f} 个词，最长 {lens.max()}；"
      f"统一截断或补齐到 {MAXLEN} 个词")
demo = ["Are you coming home for dinner tonight?", "WIN a FREE prize now! Text WIN to 80888 to claim"]
for s in demo:
    print(f"  例：“{s}”")
    print(f"   → 分词 {tok(s)}")
    print(f"   → 序列（只显示最后 12 个）{encode(s)[-12:]}")
X = lambda t: torch.tensor([encode(s) for s in t])              # noqa: E731
Xtr, Xval, Xte, Xdemo = X(txt_tr), X(txt_val), X(txt_te), X(demo)
ytr = torch.tensor(y_tr, dtype=torch.float32)
V = len(vocab) + 2


class TextRNN(nn.Module):
    """词嵌入 → 循环层 → 取最后一个时间步的隐藏状态 → 全连接输出一个数"""
    def __init__(self, cell):
        super().__init__()
        self.emb = nn.Embedding(V, 32, padding_idx=0)            # 每个词变成 32 维向量
        self.rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM}[cell](32, 32, batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))                             # h 的形状：(批大小, 40 个时间步, 32)
        return self.out(h[:, -1]).squeeze(1)                     # 读完整条短信后的隐藏状态


def report(name, pred, sec_fit, sec_pred):
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"  {name:<18}准确率 {(tp + tn) / len(pred):.3f}  精确率 {tp / (tp + fp):.3f}  召回率 {tp / (tp + fn):.3f}  "
          f"漏判 {fn:>3}  误判 {fp:>2}  训练 {sec_fit:5.2f} 秒  预测 {sec_pred * 1000:4.0f} 毫秒")


def train_torch(cell):
    torch.manual_seed(0)
    m = TextRNN(cell)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    lossf = nn.BCEWithLogitsLoss()
    t = time.perf_counter()
    log = []
    for ep in range(1, 11):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            b = perm[i:i + 64]
            loss = lossf(m(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            log.append(((m(Xval) > 0).numpy() == y_val).mean())
    sec = time.perf_counter() - t
    print(f"  {cell} 每轮验证准确率：" + " ".join(f"{a:.3f}" for a in log)
          + f"；参数 {sum(p.numel() for p in m.parameters())}")
    t = time.perf_counter()
    with torch.no_grad():
        pred = (m(Xte) > 0).numpy().astype(int)
    return m, pred, sec, time.perf_counter() - t


print("\n② PyTorch 循环神经网络（词嵌入 32 维，隐藏状态 32 维，Adam 学习率 0.003，批大小 64，10 轮）")
rnn, p_rnn, s_rnn, q_rnn = train_torch("RNN")
lstm, p_lstm, s_lstm, q_lstm = train_torch("LSTM")

print("\n③ Keras 3 写的 LSTM（结构相同）")
keras.utils.set_random_seed(0)
km = keras.Sequential([
    keras.Input(shape=(MAXLEN,)),
    keras.layers.Embedding(V, 32),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation="sigmoid"),
])
km.compile(optimizer=keras.optimizers.Adam(3e-3), loss="binary_crossentropy", metrics=["accuracy"])
t = time.perf_counter()
kh = km.fit(Xtr.numpy(), y_tr, epochs=10, batch_size=64, validation_data=(Xval.numpy(), y_val), verbose=0)
s_k = time.perf_counter() - t
t = time.perf_counter()
p_k = (km.predict(Xte.numpy(), verbose=0).ravel() > 0.5).astype(int)
q_k = time.perf_counter() - t
print("  每轮验证准确率：" + " ".join(f"{a:.3f}" for a in kh.history["val_accuracy"]) + f"；参数 {km.count_params()}")

print("\n④ 测试集对比（1673 条）")
vec = CountVectorizer()
t = time.perf_counter()
nb = MultinomialNB().fit(vec.fit_transform(pd.concat([txt_tr, txt_val])), np.r_[y_tr, y_val])
s_nb = time.perf_counter() - t
t = time.perf_counter()
p_nb = nb.predict(vec.transform(txt_te))
q_nb = time.perf_counter() - t
report("多项式朴素贝叶斯", p_nb, s_nb, q_nb)
report("简单 RNN（PyTorch）", p_rnn, s_rnn, q_rnn)
report("LSTM（PyTorch）", p_lstm, s_lstm, q_lstm)
report("LSTM（Keras 3）", p_k, s_k, q_k)

lstm.eval()
with torch.no_grad():
    pr = torch.sigmoid(lstm(Xdemo)).numpy()
print("\n  LSTM 对两条自编短信的判断：" + "；".join(f"“{s[:24]}…”垃圾概率 {p:.3f}" for s, p in zip(demo, pr)))
