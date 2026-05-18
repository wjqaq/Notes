---
type: concept
aliases: [Gemini 2.5, Gemini 2.5 Pro]
---

# Gemini 2.5 Pro

## 定义
Google 开发的大语言模型，具有先进的推理、多模态、长上下文和智能体能力。在 Entity-Rubrics 论文中用于 AbstractEdit 数据集的指令生成。

## 核心要点
1. 在 AbstractEdit 流水线中作为生成 LLM：基于 Few-shot 和随机 Persona 生成配对的抽象/显式编辑指令
2. 具备结构化输出能力（函数调用 `DiverseImageAnalysisResult_abstractExplicit`）
3. 论文中用于数据生成而非评估

## 代表工作
- Comanici et al., "Gemini 2.5: Pushing the frontier with advanced reasoning", arXiv 2025
- [[Entity-Rubrics]]: 用于 AbstractEdit 指令生成

## 相关概念
- [[Gemini 3 Flash]]
- [[AbstractEdit]]
