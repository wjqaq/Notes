---
type: concept
aliases: [Pre-norm, 前置归一化]
---

# Pre-normalization

## 定义
Transformer 架构中将归一化层（如 LayerNorm、RMSNorm）放置在子层（注意力/FFN）之前而非之后的设计模式。

## 核心要点
1. 相比 Post-norm（归一化在子层之后），Pre-norm 训练更稳定
2. 允许使用更大的学习率和更少的 warm-up 步骤
3. 配合 RMSNorm 使用是当前 LLM 训练的主流选择
4. 梯度流更顺畅，缓解深层网络训练的不稳定性

## 代表工作
- [[Qwen2]]: 使用 Pre-normalization + RMSNorm 保证训练稳定性

## 相关概念
- [[RMSNorm]]
- [[Transformer]]
