---
title: "博弈/游戏 RL 文献汇报（完整稿）"
tags: [rl, 文献综述]
date: 2026-07-28
draft: true
---

# 博弈/游戏 RL 文献汇报

来源：Zotero `RL`（不含综述/背景；不含 HLA）

# 第一部分 · LLM

**LLM 嵌入 MARL 闭环**，除了对 LLM agent 做 RL 微调。典型做法是把 LLM 放在规划器、奖励/反馈解析、通信模块等位置提升策略表现，另外部分工作（文献4–8）打开 **人类语言指导与反馈** 的接口；底层执行仍由 RL/MARL 策略完成。


| 编号  | 论文题目                                                                                                                    | 年 / 平台         | 应用领域                            | 主要内容                                                                                                  |
| --- | ----------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 1   | L2M2: A Hierarchical Framework Integrating Large Language Model and Multi-agent Reinforcement Learning                  | 2025 · IJCAI   | 协作导航                            | 分层规划；LLM 零样本高层规划生成子任务，经翻译模块分发给各 agent（CTDE）；可接预训练 MARL                                                |
| 2   | 面向智能空中博弈的大语言模型-强化学习分层决策算法                                                                                               | 2026 · 控制与决策   | 空战                              | 分层规划；LLM–RL「大脑–躯干」分层（LRHDF）；提示词迭代                                                                     |
| 3   | Long-horizon Locomotion and Manipulation on a Quadrupedal Robot with Large Language Models                              | 2025 · IROS    | 四足机器人长程任务                       | 分层规划；多 LLM（规划/参数/代码/重规划）+ 底层 RL 技能库；支持人干预重规划（单智能体，规划范式参考）                                             |
| 4   | UAM-MARL: Uncertainty-Aware Modality-Enhanced Multi-Agent Reinforcement Learning with LLM-Guided Graph Policies         | 2026 · AAMAS   | 具身多智能体（搬运/搜索等）                  | 分层规划，人类指导；LLM 规划–评判，人类语言→任务图喂给 RL agent；感知不确定度校准语言–感知落差                                               |
| 5   | Language Instructed Reinforcement Learning for Human-AI Coordination                                                    | 2023 · ICML    | 不完全信息协作（Hanabi）                 | 人类指导；InstructRL：LLM 按指令生成先验策略并约束 RL 更新幅度（含 PPO/Q）；得到偏向人类偏好的协作策略，如「我提示颜色你就出牌」                          |
| 6   | M³HF: Multi-agent Reinforcement Learning from Multi-phase Human Feedback of Mixed Quality                               | 2025 · ICML    | 协作任务（烹饪Overcooked / GRF足球）      | 人类指导；迭代：采集轨迹→人工反馈→LLM 更新奖励→权重更新                                                                       |
| 7   | LLM-Assisted Semantically Diverse Teammate Generation for Efficient Multi-agent Coordination                            | 2025 · ICML    | 协作任务（GRF等）                      | 语义队友；模型结构：共享主干+队友策略头，训练：自然语言协作行为 → 奖励 → 队友策略；推理：根据队友轨迹由大模型匹配语义相近队友策略头输出动作                             |
| 8   | Language Grounded Multi-agent Reinforcement Learning with Human-interpretable Communication                             | 2024 · NeurIPS | 异构协作任务（USAR 城市搜救，Predator Prey） | 通信模块；LM 具身自博弈，采集语言对齐数据集；训练：损失函数rl损失+通信语言对齐损失；训练后的 MARL 智能体可直接与未见过的 LLM 智能体协作，通信向量可通过余弦相似度反向翻译为人类自然语言。 |
| 9   | Leveraging Large Language Models for Effective and Explainable Multi-Agent Credit Assignment                            | 2025 · AAMAS   | 协作任务（LBF / 仓储）                  | 信用分配；LLM-MCA：大模型作为集中式 reward-critic 按个体贡献数值分解环境奖励；扩展 LLM-TACA 可向各策略显式下发中间任务                           |
| 10  | Knowing What Not to Do: Leverage Language Model Insights for Action Space Pruning in Multi-agent Reinforcement Learning | 2025 · TMLR    | 运筹控制（库存 / 交通信号）                 | 动作剪枝；eSpark：LLM 零样本生成探索函数，剪冗余状态–动作对，再据策略反馈演化改进                                                        |
| 11  | Discovering Multiagent Learning Algorithms with Large Language Models                                                   | 2026 · arXiv   | 不完全信息博弈（Poker 等）                | Deepmind：算法发现；遗传演化框架，LLM 替代传统随机变异，迭代自博弈算法配合代码骨架+提示词约束                                                 |




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




# 第三部分 · 其他人机协作

不经 LLM、也不绑定世界模型安全干预：用**人类/类人伙伴建模、跨环境零样本协调、互适应训练**等，让 RL agent 能与真人（或未见过的人类风格伙伴）协作。评测侧则以可复现的人类 proxy / 挑战赛降低真人实验成本。


| 编号  | 论文题目                                                                 | 年 / 平台        | 应用领域                     | 主要内容                                                                                          |
| --- | -------------------------------------------------------------------- | ------------- | ------------------------ | --------------------------------------------------------------------------------------------- |
| 23  | Learning to Cooperate with Humans using Generative Agents            | 2024 · NeurIPS | 协作任务（Overcooked）          | 人类伙伴建模；GAMMA：潜变量生成式人类策略，采样多样伙伴训 Cooperator；可对少量真人数据做后验偏置采样                                    |
| 24  | Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination | 2025 · ICML | 协作任务（程序化生成环境）            | 零样本协调（CEC）；跨环境分布训练学通用协作规范；无需人类数据亦可与真人协作                                                       |
| 25  | NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming | 2026 · AAMAS | 协作任务（Overcooked）          | 互适应；I-POMDP 嵌套训练，对下层自适应伙伴练上层 agent，避免共训出仅搭档专用的不透明策略                                              |
| 26  | Ad-Hoc Human-AI Coordination Challenge                               | 2025 · ICML   | 不完全信息协作（Hanabi）           | 人机评测挑战（AH2AC2）；大规模人类对局训 proxy 作可复现评测伙伴；开放有限人类数据，鼓励数据高效方法                                      |
