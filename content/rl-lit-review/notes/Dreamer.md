---
title: "Dreamer 原理解析笔记"
tags: [rl, world-model, dreamer, mbrl]
date: 2026-08-01
draft: false
zotero: "zotero://select/library/items/HTPSYNDQ"
---

# Dreamer 原理解析笔记

- **V1**：*Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020) — Hafner, Lillicrap, Ba, Norouzi
- **V2**：*Mastering Atari with Discrete World Models* (ICLR 2021) — Hafner, Lillicrap, Norouzi, Ba
- **V3**：*Mastering diverse control tasks through world models* (Nature 2025) — Hafner, Pasukonis, Ba, Lillicrap

---

## 1. 相对 Dyna：潜空间里训策略

早期 MBRL 的 **Dyna** 范式把世界模型主要当**数据生成器**：用模型 rollout 造伪转移，策略仍偏 model-free 更新。

**Dreamer** 则：先用真实交互学紧凑潜空间动力学，再在该潜空间中**想象**未来轨迹，对想象轨迹上的价值做反传以更新 actor–critic——策略学习主要发生在「脑内仿真」里，与世界模型训练**交替**进行，从而降低对真实环境步数的依赖。

---

## 2. 潜空间动力学

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

经典 Dreamer 靠 **Observation model（decoder）** 重建 $\hat{o}_t$ 来塑造潜表示；后续 **decoder-free** 世界模型（如 M3W）则去掉观测重建，改用其他信号（如预测奖励、对比/JEPA 式目标等）约束 $s_t$。

---

## 3. RSSM 数据流

RSSM（Recurrent State-Space Model）把潜状态拆成：

- $h_t$：GRU 隐状态，**确定性**地压缩历史 $(z_{<t}, a_{<t})$
- $z_t$：随机状态，**建模不确定性**（V1 为连续高斯；V3 为离散 categorical）

$q_\phi(z_t \mid h_t, x_t)$ 是**后验**：看见当前观测后对 $z_t$ 的推断，用于**训练**时编码真实轨迹。  
$p_\phi(z_t \mid h_t)$ 是**先验 / 动力学预测**：只依赖历史、不看 $x_t$，用于**想象**时开环预测下一潜状态。  
训练时用 KL 把后验与先验对齐，使「无观测也能滚」的动力学与「有观测时的编码」一致。

DreamerV3 的 RSSM（观测记 $x_t$；模型状态 $s_t \doteq \{h_t, z_t\}$）：

$$
\begin{aligned}
\text{Sequence model:} &\quad h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{Encoder:} &\quad z_t \sim q_\phi(z_t \mid h_t, x_t) \\
\text{Dynamics predictor:} &\quad \hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t) \\
\text{Reward predictor:} &\quad \hat{r}_t \sim p_\phi(\hat{r}_t \mid h_t, z_t) \\
\text{Discount predictor:} &\quad \hat{\gamma}_t \sim p_\phi(\hat{\gamma}_t \mid h_t, z_t) \\
\text{Decoder:} &\quad \hat{x}_t \sim p_\phi(\hat{x}_t \mid h_t, z_t)
\end{aligned}
$$

其中 $\hat{\gamma}_t$ 为预测折扣（终止步 $\approx 0$，否则 $\approx\gamma$）；想象时 $\lambda$-return 直接乘 $\hat{\gamma}_t$，终止处截断 bootstrap。

```
         a_{t-1}, z_{t-1}
                │
                ▼
           [ GRU ] ─────────────────────► h_t
                                           │
                    ┌──────────────────────┼──────────────────────────┐
                    ▼                      ▼                          ▼
              先验 p(z_t|h_t)        后验 q(z_t|h_t,x_t)     解码 / 奖励 / 折扣
                    │                      │                  d, r, γ (h_t,z_t)
                    │                      │                          │
                    │         x_t          │                          ▼
                    │          │           │                 x̂_t , r̂_t , γ̂_t
                    │          ▼           │
                    │     [ Encoder ]      │
                    │          │           │
                    └────►  采样 z_t  ◄────┘
                         （训练：后验；
                          想象：先验）
```

链路：**编码器 → GRU → 先验/后验 → 解码器（+ Reward + Discount predictor）**。

### 世界模型损失（DreamerV3）

损失函数=重建/预测似然 − KL。V3 将 KL **拆成双向**（KL balancing, 思路类似 [[rl-lit-review/notes/VQ-VAE#3.3 码本损失与承诺损失如何协同（为什么不可合并）|VQ-VAE 的码本/承诺损失]]），总损失为

$$
\mathcal{L}(\phi)
\doteq
\mathbb{E}_{q_\phi}\Big[
\sum_{t=1}^{T}
\big(
\beta_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}
+ \beta_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}
+ \beta_{\mathrm{rep}}\mathcal{L}_{\mathrm{rep}}
\big)
\Big]
$$

常用权重 $\beta_{\mathrm{pred}}=1$，$\beta_{\mathrm{dyn}}=0.5$，$\beta_{\mathrm{rep}}=0.1$。三项为：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{pred}}(\phi)
&\doteq
-\ln p_\phi(x_t \mid z_t, h_t)
-\ln p_\phi(r_t \mid z_t, h_t)
-\ln p_\phi(\gamma_t \mid z_t, h_t)
\\
\mathcal{L}_{\mathrm{dyn}}(\phi)
&\doteq
\max\big(
1,\
\mathrm{KL}\big[
\mathrm{sg}\big(q_\phi(z_t \mid h_t, x_t)\big)
\;\|\;
p_\phi(z_t \mid h_t)
\big]
\big)
\\
\mathcal{L}_{\mathrm{rep}}(\phi)
&\doteq
\max\big(
1,\
\mathrm{KL}\big[
q_\phi(z_t \mid h_t, x_t)
\;\|\;
\mathrm{sg}\big(p_\phi(z_t \mid h_t)\big)
\big]
\big)
\end{aligned}
$$

- $\mathcal{L}_{\mathrm{pred}}$：重建观测 + 预测奖励 + **Discount predictor** $\gamma_t$（logistic）；观测/奖励常用 symlog（或 symexp two-hot）稳定量级。
- $\mathcal{L}_{\mathrm{dyn}}$：训序列模型（先验）去贴后验；$\mathrm{sg}(q)$ 阻断表示侧梯度。
- $\mathcal{L}_{\mathrm{rep}}$：训后验变得更可预测；$\mathrm{sg}(p)$ 阻断动力学侧梯度。
- $\max(1,\cdot)$：**free bits**（约 1 nat）——KL 已够小时关掉该项，把精力留给预测损失，避免「动力学极易预测但 $z$ 不含任务信息」的退化。

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

3. 用 **Reward** 预测 $\hat{r}_\tau=\mathbb{E}[q_\theta(r_\tau \mid s_\tau)]$，用价值头 $v_\psi(s_\tau)$ 做 bootstrap。

### 4.2 $\lambda$-return

动机与 model-free（GAE / TD($\lambda$)）同族：只靠有限 $H$ 内的 $\sum\hat{r}$ 会**短视**；纯一步 TD 又**偏置大**。V1 对多种 $k$-step 回报插值，构造 $V_\lambda$（v1原文 Eq.4–6）：

$$
\begin{aligned}
V_{\mathrm{R}}(s_\tau)
&\doteq \sum_{n=\tau}^{t+H} r_n
\\
V_{\mathrm{N}}^{k}(s_\tau)
&\doteq \sum_{n=\tau}^{h-1}\gamma^{n-\tau} r_n + \gamma^{h-\tau} v_\psi(s_h),
\quad h=\min(\tau+k,\,t+H)
\\
V_\lambda(s_\tau)
&\doteq (1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1} V_{\mathrm{N}}^{n}(s_\tau)
+ \lambda^{H-1} V_{\mathrm{N}}^{H}(s_\tau)
\end{aligned}
$$

$V_{\mathrm{R}}$ 只加到 horizon、不看更远；$V_{\mathrm{N}}^{k}$ 走 $k$ 步再用 $v$ 收尾；$V_\lambda$ 对其指数加权（$\lambda\to 0$ 近 TD，$\lambda\to 1$ 近长多步；常用 $\gamma{=}0.99$，$\lambda{=}0.95$）。等价递推：

$$
V_\lambda(s_\tau)=\hat{r}_\tau+\gamma\bigl[(1-\lambda)v_\psi(s_{\tau+1})+\lambda V_\lambda(s_{\tau+1})\bigr],
\quad V_\lambda(s_{t+H})=v_\psi(s_{t+H})
$$


### 4.3 Actor / Critic 更新目标

- **Critic**： $V_\lambda$ MSE（v1）：

$$
\min_\psi \sum_{\tau=t}^{t+H}\frac{1}{2}\bigl(v_\psi(s_\tau)-V_\lambda(s_\tau)\bigr)^2
$$

- **Actor**：最大化同一套 $V_\lambda$：

$$
\max_\phi \sum_{\tau=t}^{t+H} V_\lambda(s_\tau)
$$

与 model-free（如 PPO）不同：那边环境不可微，Actor 只能用得分函数 / clip 优势；这里 Transition 可微，对上式的梯度可沿 $\phi\to a\to s'\to\hat{r},v\to V_\lambda$ **解析回传**（pathwise）。

**推理（部署）**：只需 **Representation + Actor**（$o_t\to s_t\to a_t$）；**不需要**开环想象。

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