---
title: "MuZero 原理解析笔记"
tags: [rl, world-model, muzero, mcts, mbrl]
date: 2026-08-06
draft: false
zotero: "zotero://select/library/items/HVN4QGHC"
---

# MuZero 原理解析笔记

- **MuZero**：*Mastering Atari, Go, chess and shogi by planning with a learned model* (Nature 2020) — Schrittwieser et al.
- 相对 AlphaZero：去掉对真实规则/模拟器的依赖，在学到的隐状态上做 MCTS。
- 相对 Dreamer：Dreamer 在潜空间里 actor–critic 反传；MuZero 在潜空间里 MCTS 规划，再用搜索目标训网络。

---

## 0. 蒙特卡洛树搜索（MCTS）

边 $(s,a)$ 上维护访问次数 $N(s,a)$ 与平均回报 $Q(s,a)$。每次模拟四步：

- **selection**：从根出发，反复选边直到未完全展开的叶子。经典用 **UCB1** 平衡利用与探索：

$$
a = \arg\max_{a}\left(
Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}}
\right)
$$

其中 $N(s)=\sum_{a}N(s,a)$；$N(s,a)=0$ 的边优先展开。AlphaGo / AlphaZero 改用带先验的 **PUCT**（$P$ 来自策略网络）：

$$
a = \arg\max_{a}\left(
Q(s,a) + c_{\mathrm{puct}}\, P(s,a)\,
\frac{\sqrt{N(s)}}{1+N(s,a)}
\right)
$$

- **expansion**：在叶子上展开一个子节点。经典 MCTS 用环境规则推下一步；alphago：对手按策略网络走
- **evaluation**：得到叶节点估值 $G$。经典做快速 rollout（常随机）直到终局；alphago：两个策略网络博弈（或价值网络直接估）
- **backup**：沿路径回传，更新 $N$ 与 $Q$（$G$ 为本次模拟回报）：

$$
N(s,a) \leftarrow N(s,a)+1,\qquad
Q(s,a) \leftarrow Q(s,a) + \frac{G - Q(s,a)}{N(s,a)}
$$

搜完后在根上取访问最多的动作（或按 $N$ 采样）：

$$
a^{\star} = \arg\max_{a}\, N(s_{\mathrm{root}}, a)
$$

---

## 1. AlphaGo、AlphaZero 与 MuZero

**AlphaGo**：behavior cloning → rl policy → MCTS

**AlphaZero**：MCTS 不再用于部署，而是在自博弈过程中执行，搜索得到的策略用于策略网络更新。无需 behavior cloneing 不用人类经验。

**MuZero**：AlphaZero 在 MCTS expansion 时仍需真实规则（如围棋提子）。MuZero 改为学一个世界模型拟合转移——**不重建观测、不需规则**，只在隐状态上做规划；$s$ 不必对应真实局面，只要能支撑准确的奖励、价值与策略即可。

---

## 2. 模型结构

规划时模型需要给出三类量：**策略先验** $p$（指导搜索选边）、**价值** $v$（叶节点估值，代替 rollout）、以及沿边的**即时奖励** $r$；它们都定义在隐状态 $s$ 上，不回到像素。

为此拆成三个函数。Representation $h$ 把真实观测压成根隐状态；Dynamics $g$ 在无观测条件下一步推进隐状态并预测奖励；Prediction $f$ 从隐状态读出策略与价值：

$$
s^{0} = h_{\theta}(o_{1},\ldots,o_{t})
$$

$$
r^{k},\; s^{k} = g_{\theta}(s^{k-1}, a^{k})
$$

$$
p^{k},\; v^{k} = f_{\theta}(s^{k})
$$


---

## 3. 训练目标与损失

对真实轨迹用 $g$ unroll $K$ 步，每步把对应目标当监督。

### 3.1 奖励与价值目标

$u$ 是环境真实**即时奖**，$r$ 是 $g$ 的预测；$v$ 是 $f$ 估计的**折扣回报**，$z$ 是其训练标签。对应关系：

$$
r_t^k \approx u_{t+k},\qquad
v_t^k \approx \mathbb{E}\big[u_{t+k+1}+\gamma u_{t+k+2}+\cdots\big]
$$

$z$ 与model-free思路相似 **$n$-step + 搜索价值 bootstrap**（$\nu$ 为 MCTS 价值）：

$$
z_t = u_{t+1} + \gamma u_{t+2} + \cdots + \gamma^{n-1} u_{t+n} + \gamma^{n}\nu_{t+n}
$$

棋类：$\gamma{=}1$、无中间奖，终局 $u\in\{-1,0,+1\}$，$z$ 仍由终局决定。Atari：有中间奖、$\gamma{=}0.997$，必须 bootstrap。

### 3.2 三项损失

对每个真实时刻 $t$，用真实动作序列 unroll $k=0\ldots K$，监督：

- **奖励** $l^r(u_{t+k}, r_t^k)$：动力学预测贴真实即时奖
- **价值** $l^v(z_{t+k}, v_t^k)$：预测头贴上面的 $z$
- **策略** $l^p(\pi_{t+k}, p_t^k)$：预测头贴 MCTS 的 visit 策略 $\pi$（交叉熵）

$$
\mathcal{L}_t(\theta)
=
\sum_{k=0}^{K}
\big(
l^r(u_{t+k}, r_t^k)
+ l^v(z_{t+k}, v_t^k)
+ l^p(\pi_{t+k}, p_t^k)
\big)
+ c\|\theta\|^2
$$

棋类上 $r/v$ 用平方误差；Atari 量级变化大，改用交叉熵更稳。策略项始终交叉熵。$\pi$ 与 $z$ 把规划改进蒸馏回网络；$u$ 把 $g$ 的奖励头钉在真实环境上。

---

## 4. 整体 Pipeline

三块交替：

1. **与环境交互 / 自博弈**：每步用当前网络在隐空间做一次 MCTS，按 visit 选动作。相对经典 MCTS（§0）：selection 仍用 PUCT，先验来自预测头 $p$；expansion 不再调用真实规则，而用 $g(s,a)\to(r,s')$；evaluation 直接取 $v(s)$、不再 rollout；backup 沿路径累加预测奖励与叶价值，更新 $Q,N$。最终动作与 AlphaZero 一样，按 visit count 采样或取 argmax。
2. **存经验**：$(o,a,u,\pi,z)$
3. **更新**：采样序列，$h\to g\times K\to f$，反传 §3 损失

部署时可二选一，视任务而定：用 $h{+}f$ **直接按策略** $p$ 出动作，或继续做 **隐空间 MCTS** 再按 visit 选。原文评估里，围棋等精密规划任务明显吃搜索深度；Atari 上加深搜索收益较小，甚至 **1 次模拟（等价纯策略网络）** 已表现良好——训练结束后策略往往已内化搜索带来的改进。两种方式都不需要真实规则模拟器。
