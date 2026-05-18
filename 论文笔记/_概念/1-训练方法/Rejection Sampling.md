---
type: concept
aliases: [拒绝采样, Rejection Sampling for Reasoning]
---

# Rejection Sampling

## 定义
一种数据增强策略：从模型中采样多个输出，仅保留与 ground truth 匹配的样本（拒绝不匹配的），以此构建高质量推理数据集。

## 核心要点
1. 对数学、代码等需多步推理的任务，使用中间版模型生成 CoT 输出
2. 仅保留输出与预期答案匹配的样本
3. 额外过滤 code-switching、过长、重复的输出
4. 关键挑战：视觉-语言模型的中间推理步骤可能未充分利用视觉信息
5. Qwen2.5-VL 通过规则+模型双驱动过滤验证 CoT 中间步骤的视觉整合

## 代表工作
- [[Qwen2.5-VL]]: 使用 Rejection Sampling 增强复杂推理数据
- [[DeepSeek-V3]]: 大规模 Rejection Sampling 构建训练数据

## 相关概念
- [[Chain-of-Thought]]
- [[Supervised Fine-Tuning]]
- [[Direct Preference Optimization]]
