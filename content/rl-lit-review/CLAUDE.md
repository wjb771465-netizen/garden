# rl-lit-review — 博弈/游戏 RL 文献综述工作区

本目录位于数字花园仓库内：`~/KB/garden/content/rl-lit-review/`。笔记即博客内容单一源；站点由父目录 Quartz 构建，部署至 GitHub Pages（`wjb771465-netizen.github.io/garden`）。

## Background

1–2 个月内广泛阅读 **博弈/游戏场景下的强化学习**（空战等对抗博弈为典型），**LLM 与 World Model 两条线都查**；先产出老师可读的宽清单，收窄后再写综述报告 PPT。文献用 Zotero + Better BibTeX；Cursor 经 **54yyyu/zotero-mcp**（Hybrid）读写库。本目录存放工作流说明、给老师的清单草稿、精读笔记导出与综述大纲（非 PDF 主库）。

## Key Paths

| 路径 | 用途 |
|------|------|
| `~/KB/garden/` | Quartz 工程根；`npx quartz build --serve` 本地预览 |
| `lists/` | 给老师的汇报清单草稿（按技术路线，Markdown） |
| `report.docx` | 汇报成稿（Word）；表样式以文内示例表为准 |
| `scripts/md_to_report_docx.py` | 把 `lists/*.md` 灌入 `report.docx`（克隆示例表样式） |
| `notes/` | 精读笔记导出 / 对比表草稿 |
| `outlines/` | 综述大纲 → 日后交 ppt-master |
| `~/Zotero` | 文献主库（PDF/条目） |
| `~/.cursor/mcp.json` | Zotero MCP → `zotero-mcp-hybrid` |
| `~/.local/bin/zotero-mcp-hybrid` | Local 读 + Web API 写；key 来自 `pass api/zotero` |
| `~/.claude/skills/rl-lit-review/` | （待建）Agent 流程 skill |
| `~/.cursor/plans/rl_lit_ai_workflow_b6049887.plan.md` | 原始选型 Plan |

## 博客发布（Quartz）

- 可发布 Markdown 均在 `content/` 下（含本目录 `lists/`、`notes/`）
- 每篇笔记顶部 **frontmatter 必填**：`title`、`tags`、`date`；`draft: true` 不上站
- 精读笔记建议加 `zotero: "zotero://select/library/items/<KEY>"` 回链 Zotero 条目
- 发布后可选：经 MCP 将博客 URL 回写到 Zotero 条目 `url` 字段
- `scripts/`、`*.docx`、`*.pptx`、`CLAUDE.md` 由 `quartz.config.yaml` 的 `ignorePatterns` 排除，不参与网站构建

```yaml
---
title: "笔记标题"
tags: [rl, llm]
date: 2026-07-28
draft: false
zotero: "zotero://select/library/items/XXXXXXXX"
---
```

## Zotero 空间

汇报与宽读的**主来源**是 `RL` 库（按技术路线分子集）。`Projects/RLaircombat` 等为既有主题库，不作为默认汇报源。

```
RL/                              # PJK4HARZ  ← 汇报默认来源
  LLM/                           # YVKKKA34
    llm规划器                    # FCNS48EP
    human->llm->rlagent          # 23Z627WV
  World Model/                   # BE4TJSZA
    隐式wm                       # LSZSCVMT
    显式wm                       # Q9NXPH7F
    LR                           # 6TCWTT2P  ← 综述，不进汇报清单
  Human-AI interaction           # 6NPDWKGA  ← 暂存；纯人机（非 LLM/WM）暂不汇报，视情况单开

Projects/                        # CL9LRNXC
  RLaircombat/                   # 662L9UYZ（Transformer / Selfplay / background…）
  Thesis/                        # 7WDX4FIG
```

### 标签约定（打在条目上，可选辅助）

| 维度 | 取值 |
|------|------|
| `axis:` | `llm` / `wm` / `human-ai`（后者暂不进稿） |
| `domain:` | `air-combat` / `game` / `wargame` / `robotics` |
| `status:` | `to-screen` / `skimmed` / `deep` |

## 汇报清单格式（`lists/`）

面向老师：按**技术路线**组织，不按必读/选读分层；**背景与综述不汇报**（跳过 `World Model/LR` 及 survey）。  
**纯人机协作**（不经 LLM、也不走 WM/人干预安全那条）**暂不进稿**，条目可留在 `Human-AI interaction`，后续视情况单开一类。

### 结构模板

```markdown
# 博弈/游戏 RL 文献汇报 · YYYY-MM-DD
来源：Zotero `RL`（不含综述/背景；不含纯人机暂缓项）

## 路线总览
（半页内：LLM 与 WM 两条线如何咬合；不列论文）

## 1. LLM 赋能 MARL 与人机协作
（边界：不写 LLM-as-agent / 对 LLM 做 RL 微调为主线；
 写 LLM 作规划 / 奖励与反馈解析 / 通信 / 队友语义等，打开人类指导接口）

路线导语 + **一张主表**（按角色排序、连续编号；勿拆成多张表）：

| 编号 | 论文题目 | 年 / 平台 | 应用领域 | 主要内容 |
|------|----------|-----------|----------|----------|
| 1 | 完整题名 | 2024 · ICML | 不完全信息协作（Hanabi） | 类型；技术方向；可借鉴点（末段可选） |

建议排序：规划器/分层 → 人类指导经 LLM→RL → 奖励/语义队友 → 通信 grounding →（可选）算法发现

## 2. World Model
### 2.1 隐式 WM
（导语 + 一张主表）
### 2.2 显式 WM
（导语 + 一张主表；含人干预 + MBRL 等与 WM 强绑定的条目）

## 附：待补 / 存疑
（缺发表平台、相关性未定；不写综述；不写已暂缓的纯人机）
```

灌入 Word 时用 `scripts/md_to_report_docx.py`，列与上表一致。

### 表字段

| 列 | 要求 |
|----|------|
| 编号 | 路线内从 1 递增 |
| 论文题目 | **完整题名**（非 citekey） |
| 年 / 平台 | 同一格：`YYYY · 发表平台`；平台 = **会议/期刊简称**（如 `ICML`、`NeurIPS`、`控制与决策`），不是实验环境；preprint 写 `arXiv`；未知写 `YYYY · —`，并进「待补」 |
| 应用领域 | 简要；已够具体的直接写（如 `空战`、`协作导航`、`四足机器人长程任务`）；仅宽泛标签才加括号举例（如 `不完全信息协作（Hanabi）`、`协作任务（Overcooked / 足球）`） |
| 主要内容 | 分号分段：`大致类型；技术方向；创新点/可借鉴点`；末段可不写；勿再重复应用领域，勿贴摘要全文 |

示例：`分层规划；LLM–RL 大脑–躯干分层；提示反思 + 序贯协同`

### 写法约定

- 每条技术路线：**先短导语，再一张主表**；用排序体现角色分类，勿再套「必读」标签、勿按角色拆多表
- 应用场景（空战等）与方法文可同表，场景写在「应用领域」列
- 只收录 `RL` 库内非综述、非暂缓条目；不确定标 `needs-verify`，禁止捏造

## 工作流阶段

### Phase 0 — 基建（基本完成）

- [x] 安装 `zotero-mcp-server` v0.6.1；Hybrid wrapper + `pass api/zotero`
- [x] Zotero Local API 开启；集合树建成
- [ ] `rl-lit-review` skill
- [ ] 一页日常操作说明

### Phase 1 — 宽读 → 老师汇报清单（2–4 周）

- 人：S2 / Connected Papers / arXiv → Connector 进 `RL` 对应技术路线子集
- AI：从 `RL` 导出 → `lists/` **按技术路线**成稿（见上节格式）；综述/背景/纯人机暂缓不进稿；**禁止捏造文献**
- 检索骨架：空战/博弈 + RL；再分别叠 LLM、world model / Dreamer / MBRL
- 第一部分当前块序见计划 `part1_llm_marl`（规划 → 人类指导 → 奖励/队友 → 通信 → 算法发现）

### Phase 2 — 精读与综述骨架（老师收窄后）

- 按老师圈定的路线/条目精读；笔记模板见 skill；产出进 `notes/` + `outlines/`
- PPT：大纲成熟后用 `~/Workspace/ppt-master`

## Rules

- 清单/引用只许：Zotero `RL` 库内条目，或用户刚粘贴的 DOI/arXiv；不确定标 `needs-verify`
- MCP 写库前说明将改什么；批量打标签/移动集合需用户确认意图
- 汇报按技术路线，不按必读/选读；综述、背景、纯人机（非 LLM/WM）暂不进汇报清单
- Zotero 桌面须开着才能 Local 读；写走 Web API（需联网 + pass key）
- 原始 PDF 只在 Zotero；本目录不存大二进制（组会 PPT 等可放，转换走 `~/KB/.claude/scripts`）

## Tech Stack

- Zotero 9 + Better BibTeX；MCP：`54yyyu/zotero-mcp` Hybrid
- 发现：Semantic Scholar / Connected Papers / arXiv（浏览器）
- Agent：Cursor +（待建）`rl-lit-review` skill；密钥：`pass` → `key zotero`
- 终稿：ppt-master（后续）

## 当前状态快照

- MCP 安装与集合：完成；主汇报源为 `RL/{LLM, World Model}`；`Human-AI` 暂存、纯人机不进稿
- Cursor `user-zotero`：若报错则重载 MCP（hybrid 依赖 gpg/`pass`）
- 语义检索 `[semantic]`：未装（库充实后再加）
