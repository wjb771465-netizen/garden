---
title: "第三部分 · 其他人机协作"
tags: [rl, human-ai, 文献综述]
date: 2026-07-28
draft: true
---

# 第三部分 · 其他人机协作

不经 LLM、也不绑定世界模型安全干预：用**人类/类人伙伴建模、跨环境零样本协调、互适应训练**等，让 RL agent 能与真人（或未见过的人类风格伙伴）协作。评测侧则以可复现的人类 proxy / 挑战赛降低真人实验成本。


| 编号  | 论文题目                                                                 | 年 / 平台        | 应用领域                     | 主要内容                                                                                          |
| --- | -------------------------------------------------------------------- | ------------- | ------------------------ | --------------------------------------------------------------------------------------------- |
| 23  | Learning to Cooperate with Humans using Generative Agents            | 2024 · NeurIPS | 协作任务（Overcooked）          | 人类伙伴建模；GAMMA：潜变量生成式人类策略，采样多样伙伴训 Cooperator；可对少量真人数据做后验偏置采样                                    |
| 24  | Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination | 2025 · ICML | 协作任务（程序化生成环境）            | 零样本协调（CEC）；跨环境分布训练学通用协作规范；无需人类数据亦可与真人协作                                                       |
| 25  | NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming | 2026 · AAMAS | 协作任务（Overcooked）          | 互适应；I-POMDP 嵌套训练，对下层自适应伙伴练上层 agent，避免共训出仅搭档专用的不透明策略                                              |
| 26  | Ad-Hoc Human-AI Coordination Challenge                               | 2025 · ICML   | 不完全信息协作（Hanabi）           | 人机评测挑战（AH2AC2）；大规模人类对局训 proxy 作可复现评测伙伴；开放有限人类数据，鼓励数据高效方法                                      |
