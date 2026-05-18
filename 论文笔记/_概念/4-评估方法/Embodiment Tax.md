---
aliases: [体现税, 具身税]
tags: [concept, forgetting, vla, evaluation-metric]
created: 2026-05-18
---

# Embodiment Tax（体现税）

## 定义

VLA 模型在动作微调过程中，预训练 VLM 的多模态能力被系统性侵蚀的现象。

**量化公式**：
$$
\Delta(f_{\text{VLA}}) = 1 - \frac{S(f_{\text{VLA}})}{S(f_{\text{VLM}})}
$$

其中 $S(\cdot)$ 为多模态理解基准套件的平均得分，$\Delta = 0$ 表示无损，$\Delta = 1$ 表示能力完全归零。

## 关键洞察

- 即使使用 MoT 路由（参数隔离），动作微调仍会导致 VLM 语义评分大幅下降
- MoT 的遗忘程度小于 MLP，但无法消除
- 大模型（7B）比小模型（2B）更具遗忘韧性
- 根因：单一编码器被强制同时承担语义理解（腹侧）和视觉运动控制（背侧）的功能

## 代表工作

- [[UAM]]: 提出 Embodiment Tax 概念并量化，通过 Dorsal Expert 架构分离将 tax 降至 <5%
- [[VLM4VLA]]: 系统研究 VLA 中表征瓶颈和遗忘问题
