---
title: "Dreamer 原理解析笔记"
tags: [rl, world-model, dreamer, mbrl]
date: 2026-08-01
draft: false
zotero: "zotero://select/library/items/HTPSYNDQ"
---

# Dreamer 原理解析笔记

- **论文**：*Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020)
- **作者**：Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi

---

## 1. 相对 Dyna：潜空间里训策略

早期 MBRL 的 **Dyna** 范式把世界模型主要当**数据生成器**：用模型 rollout 造伪转移，策略仍偏 model-free 更新。

**Dreamer** 则：先用真实交互学紧凑潜空间动力学，再在该潜空间中**想象**未来轨迹，对想象轨迹上的价值做反传以更新 actor–critic——策略学习主要发生在「脑内仿真」里，与世界模型训练**交替**进行，从而降低对真实环境步数的依赖。

---

## 2. 潜空间动力学：四件套

潜状态 $s_t$，世界模型参数 $\theta$。

**Representation model**：

$$
p_\theta(s_t \mid s_{t-1}, a_{t-1}, o_t)
$$

**Observation model**：

$$
q_\theta(o_t \mid s_t)
$$

**Reward model**：

$$
q_\theta(r_t \mid s_t)
$$

**Transition model**：

$$
q_\theta(s_t \mid s_{t-1}, a_{t-1})
$$

---

## 3. RSSM 数据流

RSSM（Recurrent State-Space Model）把潜状态拆成：

- $h_t$：GRU 隐状态，**确定性**地压缩历史 $(z_{<t}, a_{<t})$
- $z_t$：随机状态，**建模不确定性**（V1 为连续高斯）

$q_\phi(z_t \mid h_t, o_t)$ 是**后验**：看见当前观测后对 $z_t$ 的推断，用于**训练**时编码真实轨迹。  
$p_\phi(z_t \mid h_t)$ 是**先验**：只依赖历史、不看 $o_t$，用于**想象**时开环预测下一潜状态。  
训练时用 KL 把后验拉向先验，使「无观测也能滚」的动力学与「有观测时的编码」一致。

```
         a_{t-1}, z_{t-1}
                │
                ▼
           [ GRU ] ─────────────────────► h_t
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
              先验 p(z_t|h_t)        后验 q(z_t|h_t,o_t)      解码 / 奖励
                    │                      │                   d(h_t,z_t)
                    │                      │                   r(h_t,z_t)
                    │         o_t          │                      │
                    │          │           │                      ▼
                    │          ▼           │                 ô_t , r̂_t
                    │     [ Encoder ]      │
                    │          │           │
                    └────►  采样 z_t  ◄────┘
                         （训练：后验；
                          想象：先验）
```

链路：**编码器 → GRU → 先验/后验 → 解码器（+ 奖励头）**。

### ELBO 原理

目标是最大化观测（及奖励）的对数似然 $\log p(o_{1:T} \mid a_{1:T})$，但边缘化全部潜变量 $z_{1:T}$ 不可行。引入后验近似 $q(z \mid h, o)$ 后，由 Jensen / 变分可得 **ELBO**（证据下界）：

$$
\log p(o) \ge
\underbrace{\mathbb{E}*{q(z\mid h,o)}\big[\log p(o \mid h,z)\big]}*{\text{重建项：用 }z\text{ 解释观测}}
-
\underbrace{\mathrm{KL}\big(q(z \mid h,o)  p(z \mid h)\big)}_{\text{复杂度项：后验勿偏离先验}}
$$

- **重建项**大 → $z$ 携带足够信息，解码器能还原 $o$（奖励头同理，可并进似然）。
- **KL 项**小 → 后验接近「不看 $o$ 也能预测」的先验，想象时用 $p$ 采样才靠谱；过大则后验偷懒忽略观测，过小或后验过强则易**后验坍塌**。

实现上最小化负 ELBO，即：

$$
\mathcal{L}
= \mathcal{L}_{\mathrm{recon}}(o_t, \hat{o}_t) + \mathcal{L}_r(r_t, \hat{r}_t)

- \mathrm{KL}\big(q_\phi(z_t \mid h_t, o_t)  p_\phi(z_t \mid h_t)\big)
$$

---

## 4. 策略优化与推理

动力学已知且可微时，行为学习在潜空间**想象**中完成，与世界模型训练交替进行——无需在真实环境逐步试错更新策略。

### 4.1 想象轨迹

1. 从 replay 取真实片段，用 **Representation** $p_\theta(s_t \mid s_{t-1}, a_{t-1}, o_t)$ 得到起点 $s_t$（有观测）。
2. 往后滚 $H$ 步，**只用 Transition + Actor**，不再看 $o$：

$$
a_\tau \sim q_\phi(a_\tau \mid s_\tau),\qquad
s_{\tau+1} \sim q_\theta(s_{\tau+1} \mid s_\tau, a_\tau)
$$

1. 用 **Reward** 预测 $\hat{r}*\tau=\mathbb{E}[q*\theta(r_\tau \mid s_\tau)]$，用价值头 $v_\psi(s_\tau)$ 做 bootstrap。

### 4.2 $\lambda$-return

动机与 model-free（GAE / TD($\lambda$)）同族：只靠有限 $H$ 内的 $\sum\hat{r}$ 会**短视**；纯一步 TD（$\hat{r}*\tau+\gamma v*\psi(s_{\tau+1})$）又**偏置大**。因此对多种 $N$-step 回报插值，构造 $V_\lambda$。

**$N$-step return**（走 $N$ 步再用 $v$ 收尾）：

$$
V_N(s_\tau)
= \hat{r}*\tau + \gamma\hat{r}*{\tau+1} + \cdots + \gamma^{N-1}\hat{r}*{\tau+N-1} + \gamma^N v*\psi(s_{\tau+N})
$$

**$\lambda$-return**（原文 Eq.6）：

$$
V_\lambda(s_\tau)
= (1-\lambda)\sum_{N=1}^{H-1}\lambda^{N-1} V_N(s_\tau)

- \lambda^{H-1} V_H(s_\tau)
$$

$\lambda\to 0$ 近乎纯 TD，$\lambda\to 1$ 近乎长多步回报（常用 $\sim 0.95$）。递推写法：

$$
V_\lambda(s_\tau)=\hat{r}*\tau+\gamma\bigl[(1-\lambda)v*\psi(s_{\tau+1})+\lambdaV_\lambda(s_{\tau+1})\bigr]
$$

末端 $V_\lambda(s_{t+H})=v_\psi(s_{t+H})$。$V_\lambda$ **不是网络**，而是用 $\hat{r}$ 与 $v_\psi$ 算出的回报标签（来自想象轨迹，非真环境）。

### 4.3 Actor / Critic 更新目标

- **Critic**：把 $V_\lambda$ 当老师，回归拟合（算 $V_\lambda$ 时其中的 $v$ 常 `stopgrad`，避免目标随 $\psi$ 漂）：

$$
\min_\psi \sum_{\tau=t}^{t+H}\frac{1}{2}\bigv_\psi(s_\tau)-V_\lambda(s_\tau)\big^2
$$

- **Actor**：最大化同一套 $V_\lambda$：

$$
\max_\phi \sum_{\tau=t}^{t+H} V_\lambda(s_\tau)
$$

与 model-free（如 PPO）不同：那边环境不可微，Actor 只能用得分函数 / clip 优势；这里 Transition 可微，对上式的梯度可沿 $\phi\to a\to s'\to\hat{r},v\to V_\lambda$ **解析回传**（pathwise）。对世界模型 $\theta$ 常 `stopgrad`，只更新 $\phi,\psi$，避免策略把动力学「骗」成虚高回报。

**推理（部署）**：只需 **Representation + Actor**（$o_t\to s_t\to a_t$）；**不需要**开环想象，也不必跑 Critic。

---

## 5. Dreamer V1 整体 Pipeline（Algorithm 1）

原文记号：表征用 $p_\theta$（见观测），转移/奖励/观测解码用 $q_\theta$；动作 $q_\phi$，价值 $v_\psi$。与上文「后验 $q$ / 先验 $p$」的教学习惯**字母相反**，以下伪代码跟论文。

**Algorithm 1: Dreamer**

1. Initialize dataset $\mathcal{D}$ with $S$ random seed episodes
2. Initialize parameters $\theta, \phi, \psi$ randomly
3. **while** not converged **do**
  1. **for** update step $c = 1,\ldots,C$ **do**
    - *Dynamics learning*
      - Draw $B$ sequences $(a_t, o_t, r_t)_{t=k}^{k+L} \sim \mathcal{D}$
      - Compute model states $s_t \sim p_\theta(s_t \mid s_{t-1}, a_{t-1}, o_t)$
      - Update $\theta$ via representation learning（重建 $o$、预测 $r$、KL 等）
    - *Behavior learning（latent imagination）*
      - Imagine trajectories $(s_\tau, a_\tau)_{\tau=t}^{t+H}$ from each $s_t$
        - $a_\tau \sim q_\phi(a_\tau \mid s_\tau)$
        - $s_{\tau+1} \sim q_\theta(s_{\tau+1} \mid s_\tau, a_\tau)$（只用转移，不看 $o$）
      - Predict rewards $\mathbb{E}[q_\theta(r_\tau \mid s_\tau)]$ and values $v_\psi(s_\tau)$
      - Compute value estimates $V_\lambda(s_\tau)$（$\lambda$-return，原文 Eq.6）
      - Update $\phi \leftarrow \phi + \alpha \nabla_\phi \sum_{\tau=t}^{t+H} V_\lambda(s_\tau)$（最大化想象价值）
      - Update $\psi \leftarrow \psi - \alpha \nabla_\psi \sum_{\tau=t}^{t+H} \frac{1}{2}v_\psi(s_\tau) - V_\lambda(s_\tau)^2$
  2. *Environment interaction*
    - $o_1 \leftarrow \mathrm{env.reset}()$
    - **for** time step $t = 1,\ldots,T$ **do**
      - Compute $s_t \sim p_\theta(s_t \mid s_{t-1}, a_{t-1}, o_t)$ from history
      - Compute $a_t \sim q_\phi(a_t \mid s_t)$
      - Add exploration noise to action
      - $r_t, o_{t+1} \leftarrow \mathrm{env.step}(a_t)$
    - Add experience $\mathcal{D} \leftarrow \mathcal{D} \cup (o_t, a_t, r_t)_{t=1}^{T}$

超参（原文）：$S$ seed episodes，$C$ collect interval，$B$ batch size，$L$ sequence length，$H$ imagination horizon，$\alpha$ learning rate。

三块交替：**学动力学（真实数据）→ 在潜空间想象里学行为 → 与环境交互补数据**。