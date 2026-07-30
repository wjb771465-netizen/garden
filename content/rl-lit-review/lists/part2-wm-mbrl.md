---
title: "第二部分 · World Model / MBRL"
tags: [rl, world-model, mbrl, 文献综述]
date: 2026-07-28
draft: false
---

# 第二部分 · World Model / MBRL

相对**无模型 RL**策略直接在真实环境试错中更新策略，**基于模型的强化学习（MBRL）**先利用交互数据学习动力学（世界模型），再在模型上规划或想象 rollout。训练方面世界模型rollout无需真实交互或复杂仿真，大大提高了样本效率。推理时可利用世界模型做规划推演，提高策略表现。

发展上，早期 Dyna 等把规划嵌入学习循环；深度时代出现可从高维观测学习的世界模型，代表性工作为 DeepMind 的 **Dreamer 系列**：先用真实交互学一个紧凑潜空间动力学模型，再在该潜空间中想象未来轨迹，并对想象轨迹上的价值反传梯度以更新策略——策略学习主要发生在脑内仿真里，从而大幅降低对真实环境步数的依赖。


| 编号  | 论文题目                                                                                    | 年 / 平台          | 应用领域                              | 主要内容                                                                                   |
| --- | --------------------------------------------------------------------------------------- | --------------- | --------------------------------- | -------------------------------------------------------------------------------------- |
| 12  | CoDreamer: Communication-Based Decentralised World Models                               | 2024 · CoCoMARL | 多智能体协作                            | Dreamer 多智能体扩展；GNN 双层通信分别增强世界模型与策略。                                                    |
| 13  | Aligning Credit for Multi-Agent Cooperation via Model-based Counterfactual Imagination  | 2024 · AAMAS    | 协作多智能体控制                          | Dreamer 多智能体扩展：MACD；中心想象–去中心执行（CIDE）生成伪数据；世界模型反事实轨迹：真实轨迹与agent零动作轨迹差值，缓解信用分配           |
| 14  | Learning and Planning Multi-Agent Tasks via an MoE-based World Model                    | 2025 · NeurIPS  | 多任务多智能体控制（Bi-DexHands / MAMuJoCo） | MoE 世界模型（SoftMoE 动力学 + SparseMoE 奖励）                                                   |
| 15  | Grounded Answers for Multi-agent Decision-making Problem through Generative World Model | 2024 · NeurIPS  | 星际争霸多智能体（SMAC）                    | VQ-VAE 图像分词器：将高维游戏图像压缩为离散码本 token 序列→动力学模型+奖励模型训练 RL；VisionSMAC 数据集：解析数值回放文件，用于训练世界模型 |
| 16  | Episodic Future Thinking Mechanism for Multi-agent Reinforcement Learning               | 2024 · NeurIPS  | 多智能体驾驶 / 粒子环境                     | 用奖励权重向量定义智能体 “角色”（行为偏好），真实观测→角色推断→对手行为预测→虚拟未来观测→生成当前执行动作；情节式未来思维提升异质智能体交互收益            |
| 17  | Acting Beyond Learning: Imagination-Assisted Decision-Making in the Visual-based Multi-Agent Cooperative Scenarios | 2025 · AAAI | 视觉多智能体协作（PettingZoo） | 潜空间世界模型（CLWPO）；对比变分界优化潜动力学 + 启发式策略优化（模型自由学习与模型规划结合）；队友模型队列与自适应 rollout |
| 18  | Look Before You Leap: Safe Model-Based Reinforcement Learning with Human Intervention   | 2022 · CoRL     | 安全控制                              | 安全 MBRL（MBHI）；动力学模型想象轨迹预判灾难态 + 模仿人类阻断；遇险则 MPC 输出安全策略                                   |


与上不同：这类工作**不独立建**能想象、能重建观测的世界模型，而是学一套主要为**价值与规划**服务的潜动力学——对不对得上真实画面不是目标。典型如 **MuZero**：交互数据只用来训练「当前隐状态 + 动作 → 下一隐状态、奖励、价值」；推理时在这套隐状态上做 MCTS，不回到像素。**VIN** 则把值迭代直接嵌进网络里做可微规划。多智能体延伸见下表。


| 编号  | 论文题目                                                                        | 年 / 平台         | 应用领域                      | 主要内容                                                              |
| --- | --------------------------------------------------------------------------- | -------------- | ------------------------- | ----------------------------------------------------------------- |
| 19  | Multi-Agent Routing Value Iteration Network                                 | 2020 · ICML    | 多智能体路由                    | 将值迭代结构嵌入网络，在图/路由任务上做可微分规划式推理                                      |
| 20  | Efficient Multi-agent Reinforcement Learning by Planning                    | 2024 · ICLR    | 星际争霸多智能体（SMAC）            | MAZero；中心化学模型 + MCTS；OS(λ) 与 AWPO 提升大联合动作空间下的搜索效率                 |
| 21  | Multiagent Gumbel MuZero: Efficient Planning in Combinatorial Action Spaces | 2024 · AAAI    | 协作多智能体控制                  | Gumbel 采样扩展 MuZero 至组合/指数动作空间，低模拟预算下仍可改进策略                        |
| 22  | MALinZero: Efficient Low-Dimensional Search for Mastering Complex Multi-Agent Planning | 2025 · NeurIPS | 矩阵博弈 / 星际争霸（SMAC / SMACv2） | 联合回报压成低维线性表示 + LinUCT 做 MCTS；缓解多智能体组合动作空间爆炸                       |


