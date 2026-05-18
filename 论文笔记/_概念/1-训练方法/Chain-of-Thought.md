---
type: concept
aliases: [思维链, CoT, Chain of Thought]
---

# Chain-of-Thought

## 定义
一种推理范式，模型在给出最终答案前显式生成中间推理步骤的过程，分为 Non-thinking（直接回答）和 Thinking（CoT 推理后回答）两种模式。

## 核心要点
1. Thinking 模式通过延长推理时间来提升复杂问题的准确率
2. Qwen3-VL 在 SFT 和 RL 阶段均对 Thinking 数据做了精细的冷启动和过滤
3. Long-CoT 数据通过拒绝采样和难度筛选来确保推理质量

## 代表工作
- [[Qwen3-VL]]: Thinking 变体在数学推理和视觉谜题上显著优于 Instruct 变体

## 相关概念
- [[SAPO]]
- [[Knowledge Distillation]]
