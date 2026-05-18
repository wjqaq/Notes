---
type: concept
aliases: [平方根重加权, Square Root Reweighting]
---

# Square-Root Reweighting

## 定义
一种训练损失重加权策略，将每个样本的损失按 token 数的平方根归一化，替代传统 per-sample 平均损失，以平衡文本和多模态数据对梯度的贡献。

## 数学形式

$$
\mathcal{L} = \sum_{i} \frac{\mathcal{L}_i}{\sqrt{N_i}}
$$

**符号说明**:
- $\mathcal{L}_i$: 第 i 个样本的损失
- $N_i$: 该样本的 token 数量

## 核心要点
1. 防止长序列样本（如长文档、长视频）主导训练梯度
2. 平衡纯文本数据和视觉-语言数据的贡献
3. 提升多模态性能同时不损害文本能力

## 代表工作
- [[Qwen3-VL]]: 首次使用平方根重加权损失

## 相关概念
- [[Cross-Entropy Loss]]
