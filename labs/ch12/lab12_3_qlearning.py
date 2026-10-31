"""实验 12-3 Q 学习走冰湖

Gymnasium 的冰湖环境（FrozenLake，4×4）：从左上角 S 出发，走到右下角 G 得 1 分，掉进冰窟窿 H 回合结束、得 0 分。
① 不打滑的冰湖：从零写 Q 学习，看 Q 表怎样一步步学出来，画出最终策略；
② 不探索（ε = 0）会怎样；
③ 打滑的冰湖（往一个方向走，只有三分之一的机会走对）：Q 学习的策略与随机乱走比较。
运行：python lab12_3_qlearning.py（在 labs/ch12 目录下）
"""
import logging

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
ARROW = "←↓→↑"                                   # 动作 0 左、1 下、2 右、3 上


def q_learning(env, episodes, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.01, seed=0, random_tie=True):
    """Q 学习：Q(s,a) ← Q(s,a) + α [r + γ max Q(s',·) − Q(s,a)]"""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards, first = [], None
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (0.8 * episodes))   # 探索率逐渐减小
        s, _ = env.reset(seed=int(rng.integers(1e9)))
        done, total = False, 0
        while not done:
            if rng.random() < eps:                                   # 探索：随机试一个动作
                a = int(rng.integers(4))
            else:                                                    # 利用：选 Q 值最大的动作（一样大时随机选）
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max()))) if random_tie else int(Q[s].argmax())
            s2, r, term, trunc, _ = env.step(a)
            target = r + (0 if term else gamma * Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            s, done, total = s2, term or trunc, total + r
        rewards.append(total)
        if total > 0 and first is None:
            first = ep + 1
    return Q, np.array(rewards), first


def evaluate(env, policy, n=1000, seed=123):
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        s, _ = env.reset(seed=int(rng.integers(1e9)))
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(int(policy(s)))
            done = term or trunc
        wins += r > 0
    return wins / n


def show(Q, desc):
    rows = []
    for i in range(4):
        row = ""
        for j in range(4):
            c = desc[i][j].decode()
            row += f" {c if c in 'HG' else ARROW[int(Q[i * 4 + j].argmax())]}"
        rows.append(row)
    return rows


# ① 不打滑
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
desc = env.unwrapped.desc
print("冰湖地图（S 起点，F 冰面，H 冰窟窿，G 终点）：")
for r in desc:
    print("   " + " ".join(c.decode() for c in r))
print(f"状态 {env.observation_space.n} 个（格子编号 0～15），动作 {env.action_space.n} 个（0 左，1 下，2 右，3 上）")

Q, R, first = q_learning(env, 2000)
print("\n① 不打滑的冰湖，Q 学习 2000 回合（α = 0.1，γ = 0.95，ε 从 1 逐渐降到 0.01）")
print(f"  第一次走到终点：第 {first} 回合")
print("  每 200 回合的成功率：" + " ".join(f"{R[i:i + 200].mean():.2f}" for i in range(0, 2000, 200)))
print("  起点的 Q 值（左、下、右、上）：" + "，".join(f"{v:.3f}" for v in Q[0]))
print(f"  理论最优：最短 6 步到终点，起点最优 Q 值 = γ⁵ = 0.95⁵ = {0.95 ** 5:.3f}")
print("  学到的策略（每格 Q 值最大的动作）：")
for row in show(Q, desc):
    print("   " + row)
s, _ = env.reset(seed=0)
path, done = [s], False
while not done:
    s, r, term, trunc, _ = env.step(int(Q[s].argmax())); path.append(s); done = term or trunc
print(f"  按策略走一遍：{' → '.join(map(str, path))}，{len(path) - 1} 步，奖励 {r:.0f}")
print("  Q 表（行是格子 0～15，列是左、下、右、上）：")
for i in range(16):
    print(f"    {i:>2} " + " ".join(f"{v:.3f}" for v in Q[i]))

# ② 不探索
print("\n② 不探索（ε = 0），各 2000 回合")
Q0, R0, _ = q_learning(env, 2000, eps_start=0.0, eps_end=0.0, random_tie=False)
print(f"  Q 值一样大时总选第一个动作（左）：成功 {int(R0.sum())} 次，Q 表非零元素 {int((Q0 != 0).sum())} 个"
      f"——在起点一直往左撞墙，到 100 步上限才结束")
Q1, R1, first1 = q_learning(env, 2000, eps_start=0.0, eps_end=0.0)
print(f"  Q 值一样大时随机选：第 {first1} 回合碰巧走到终点，之后成功 {int(R1.sum())} 次，但 Q 表非零元素只有 {int((Q1 != 0).sum())} 个"
      f"（①中为 {int((Q != 0).sum())} 个）——只认得碰巧找到的那一条路")
# ③ 打滑
envs = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
Qs, Rs, _ = q_learning(envs, 20000, alpha=0.05, gamma=0.99)
envs.action_space.seed(0)
rand_rate = evaluate(envs, lambda s: envs.action_space.sample())
q_rate = evaluate(envs, lambda s: Qs[s].argmax())
print("\n③ 打滑的冰湖，Q 学习 20000 回合（α = 0.05，γ = 0.99）")
print("  训练中每 2000 回合的成功率：" + " ".join(f"{Rs[i:i + 2000].mean():.2f}" for i in range(0, 20000, 2000)))
print(f"  测试 1000 回合：随机乱走成功率 {rand_rate:.3f}，按学到的策略走成功率 {q_rate:.3f}")
print("  学到的策略：")
for row in show(Qs, desc):
    print("   " + row)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
k = 50
ax[0].plot(np.convolve(R, np.ones(k) / k, "valid"), label="不打滑，ε 递减")
ax[0].plot(np.convolve(R1, np.ones(k) / k, "valid"), label="不打滑，ε = 0，Q 值相同时随机选")
ax[0].set_xlabel("回合"); ax[0].set_ylabel(f"成功率（{k} 回合滑动平均）"); ax[0].set_title("① ② 奖励曲线"); ax[0].legend(fontsize=8)
k2 = 500
ax[1].plot(np.convolve(Rs, np.ones(k2) / k2, "valid"))
ax[1].set_xlabel("回合"); ax[1].set_title(f"③ 打滑的冰湖（{k2} 回合滑动平均）")
V = Q.max(1).reshape(4, 4)
ax[2].imshow(V, cmap="Blues")
for i in range(4):
    for j in range(4):
        c = desc[i][j].decode()
        txt = c if c in "HG" else f"{ARROW[int(Q[i * 4 + j].argmax())]}\n{V[i, j]:.2f}"
        ax[2].text(j, i, txt, ha="center", va="center", fontsize=12, color="crimson" if c == "H" else "k")
ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].set_title("① 学到的策略和每格最大 Q 值")
for a in ax[:2]:
    a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("lab12_3_qlearning.png", dpi=90)
