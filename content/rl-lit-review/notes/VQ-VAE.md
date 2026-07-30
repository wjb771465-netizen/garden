---
title: "VQ-VAE 原理解析与应用笔记"
tags: [rl, world-model, vq-vae, generative-ai]
date: 2026-07-30
draft: false
zotero: "zotero://select/library/collections/PJK4HARZ"
---

# VQ-VAE 原理解析与应用笔记

* **原论文**：*Neural Discrete Representation Learning* (NeurIPS 2017)
* **作者**：Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu

---

## 1. 演进脉络：AE → VAE → VQ-VAE

1. **自编码器 (AE)**：
   * **流程**：图像 → 编码器 → 确定性连续向量 $z$ → 解码器 → 图像
   * **特点与问题**：隐空间无结构与概率分布约束，容易过拟合，缺乏生成能力（无法在隐空间随机采样生成新样本）。
2. **变分自编码器 (VAE)**：
   * **流程**：图像 → 编码器 → 高斯连续分布 $(\mu, \sigma)$ → 重参数化采样连续向量 $z \sim \mathcal{N}(\mu, \sigma^2)$ → 解码器 → 图像
   * **特点与问题**：引入高斯连续分布与 KL 散度约束，具备连续概率采样能力；但容易发生**后验坍塌（Posterior Collapse）**导致生成图像偏模糊，且**连续向量不易直接接入自回归 Transformer**。
3. **向量量化变分自编码器 (VQ-VAE)**：
   * **流程**：图像 → 编码器 → 连续特征图 $z_e$ → 向量量化 (Codebook) → 离散特征 $z_q$ → 解码器 → 图像
   * **特点与优势**：隐空间采用**离散分布**，彻底避免了后验坍塌；隐空间离散 Codebook 类似 NLP 中的**词嵌入（Word Embeddings）**，将高维视觉观测转换为离散 Visual Tokens，天然契合 Transformer 架构。

---

## 2. 核心 Pipeline

```
输入图像 / 观测 x
       │
       ▼
 [ CNN 编码器 ]
       │
       ▼
 连续特征图 z_e(x)  (形状: H × W × D)
       │
       ├─────────────────────────────────┐
       ▼                                 │
[ 离散码本 Codebook ]                   │
 类似 NLP 词嵌入 (K 个 D 维向量)        │
       │                                 │
       ▼ (取最近邻 / argmin)              │
 离散索引序列 k & 离散特征图 z_q(x)      │ (STE 梯度直通)
       │                                 │
       ▼                                 │
 [ CNN 解码器 ] ◄────────────────────────┘
       │
       ▼
 重建图像 / 观测 x̂
```

1. **编码 (Encode)**：图像 $x$ 经过 CNN 编码器输出连续特征图 $z_e(x) \in \mathbb{R}^{H \times W \times D}$。
2. **向量量化 (Vector Quantization)**：
   * 维护一个包含 $K$ 个 $D$ 维向量的码本（Codebook $e \in \mathbb{R}^{K \times D}$，作用类似 NLP 的词表/词嵌入）。
   * 对于 $z_e(x)$ 中的每个空间位置向量，在 Codebook 中找到距离最近的码本向量：

$$
z_q(x)_i = e_k \quad \text{where } k = \arg\min_j \|z_e(x)_i - e_j\|_2
$$

3. **解码 (Decode)**：将量化后的离散特征图 $z_q(x)$ 传入 CNN 解码器，重建图像或观测 $\hat{x} = D(z_q(x))$。

---

## 3. 训练机制：STE 与损失函数

### 3.1 直通估计器 (Straight-Through Estimator, STE)
量化过程中的 $\arg\min$ 操作是非连续且不可导的（梯度的导数处处为 0）。为使 Encoder 能够接受反向传播梯度，引入 **STE 技巧**：
* **前向传播**：使用离散量化后的 $z_q(x)$ 传给 Decoder。
* **反向传播**：将量化算子视为恒等映射，把 Decoder 传给 $z_q$ 的梯度**原封不动地直接传回**给连续特征 $z_e$。

**工程实现（PyTorch 技巧）**：

$$
z_q^{\text{STE}} = z_e + \text{sg}[z_q - z_e]
$$

*（其中 $\text{sg}[\cdot]$ 表示 `detach()` / Stop Gradient 操作。前向数值等于 $z_q$，反向传播时对 $z_e$ 的导数为 1）*

### 3.2 损失函数公式 (Loss Functions)

VQ-VAE 整体损失由三部分相加而成：

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{recon}}(x, D(z_q(x)))}_{\text{Reconstruction Loss}} + \underbrace{\|\text{sg}[z_e(x)] - z_q(x)\|_2^2}_{\text{Codebook Loss}} + \underbrace{\beta \|\text{sg}[z_q(x)] - z_e(x)\|_2^2}_{\text{Commitment Loss}}
$$

1. **重建损失 (Reconstruction Loss, $\mathcal{L}_{\text{recon}}$)**：如 MSE 或 L1，同时优化编码器与解码器，使重建输出 $\hat{x}$ 尽可能接近原图 $x$。
2. **码本损失 (Codebook Loss)**：使用 L2 损失将选中的码本向量 $z_q$ 拉向编码器输出 $z_e$（实际工程中也常用指数移动平均 EMA 更新码本）。
3. **承诺损失 (Commitment Loss)**：权重 $\beta$ 通常设为 0.25，防止编码器输出 $z_e$ 波动过大，使其“承诺”稳定在当前选中的码本向量附近。

### 3.3 码本损失与承诺损失如何协同（为什么不可合并）

两项前向数值相同（都是 $\|z_e - z_q\|_2^2$），区别只在 `sg[·]`（stop-gradient，即 `detach()`：前向原样通过，反向梯度断流，被包住的变量求导时视为常数）的位置——它决定梯度流向谁：码本损失冻结 $z_e$、只更新码本（**码本追编码器**）；承诺损失冻结 $z_q$、只更新编码器（**编码器向码本承诺**）。缺了承诺损失，$z_e$ 会在码本向量间漂移、最近邻频繁跳变导致训练失稳；$\beta = 0.25 < 1$ 是让码本先学快、形成稳定锚点，编码器再小步贴近。工程上常用 EMA 统计更新码本替代码本损失，此时损失只剩重建 + 承诺两项。

---

## 4. 在世界模型 (World Model) 与 RL 中的应用

1. **Tokenizer (视觉分词器)**：
   * 编码器作为 Tokenizer，将环境高维视觉观测/状态压缩为离散 Token 序列，送入世界模型的 Transformer Backbone 进行自回归预测。
2. **决策与评估 (Actor/Critic)**：
   * 解码器用于重建观测以进行想象/可视化；离散表征与世界模型预测的隐状态被送入 Actor/Critic 网络进行策略学习（如 MARIN、LBI 等多智能体/生成式世界模型）。
