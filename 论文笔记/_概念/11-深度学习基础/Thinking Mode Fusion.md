---
type: concept
aliases: [Thinking Mode Fusion]
---

# Thinking Mode Fusion

## 定义
Qwen3 后训练第三阶段，在已有 thinking 能力的模型上通过 SFT 融合 non-thinking 模式，使单一模型能动态切换两种推理模式，并自然涌现 Thinking Budget 能力。

## 核心要点
1. SFT 数据包含 thinking 和 non-thinking 两类数据
2. Chat Template 使用 `/think` 和 `/no_think` 标志控制模式
3. Non-thinking 回复保留空 `<think></think>` 块以保持格式一致性
4. 默认 thinking 模式（不指定时），多轮对话遵循最后一个标志
5. Thinking Budget 能力是此阶段的自然涌现，非显式训练

## 代表工作
- [[Qwen3]]: 首次提出 Thinking Mode Fusion

## 相关概念
- [[Thinking Budget]]
- [[Chain of Thought]]
- [[GRPO]]
- [[Strong-to-Weak Distillation]]
