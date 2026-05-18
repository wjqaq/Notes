---
type: concept
aliases: [Qwen2.5-Math, Qwen Math]
---

# Qwen2.5-Math

## 定义
Qwen 团队发布的数学专用大语言模型，通过自我改进（self-improvement）策略达到专家级数学推理能力。其预训练数据和 CoT SFT 数据被 Qwen2.5 复用。

## 核心要点
1. 为 Qwen2.5 提供数学预训练数据和链式推理 SFT 数据
2. 使用拒绝采样 + 奖励模型 + 标注答案指导
3. 数据质量通过 Qwen2-Math-RM-72B 过滤

## 代表工作
- [[Qwen2.5]]: 复用 Qwen2.5-Math 的预训练和 SFT 数据
- Yang et al. (2024b): 原始技术报告

## 相关概念
- [[Qwen2.5-Coder]]
- [[Chain-of-Thought|思维链]]
- [[Rejection Sampling|拒绝采样]]
