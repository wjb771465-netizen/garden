---
title: "第一部分 · LLM 赋能 MARL"
tags: [rl, llm, marl, 文献综述]
date: 2026-07-28
draft: false
---

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


