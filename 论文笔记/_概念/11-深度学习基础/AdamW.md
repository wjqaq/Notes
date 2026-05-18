---
type: concept
aliases: [AdamW优化器]
---

# AdamW

## 定义
Adam 优化器的变体，将权重衰减（weight decay）与梯度更新解耦，在深度学习尤其是 Transformer 训练中广泛使用。

## 核心要点
1. 与原始 Adam 的 L2 正则化不同，AdamW 直接在权重上施加 decay
2. 在 LLM 训练中通常设置 $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$
3. 配合 cosine learning rate schedule 和 warmup 步骤使用

## 代表工作
- [[Qwen]]: 预训练、SFT、RLHF、代码继续预训练等所有阶段均使用 AdamW
- [[LLaMA]] / [[LLaMA 2]]: 使用 AdamW 训练

## 相关概念
- [[Transformer]]
- [[Supervised Fine-Tuning]]
