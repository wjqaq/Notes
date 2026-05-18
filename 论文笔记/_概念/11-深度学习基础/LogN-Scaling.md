---
type: concept
aliases: [LogN注意力缩放, LogN Attn Scaling]
---

# LogN-Scaling

## 定义
一种训练无关的注意力缩放方法，根据上下文长度与训练长度的比值对 Q-K 点积进行缩放，确保注意力熵在长序列时保持稳定。

## 数学形式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k} \cdot \frac{\log n}{\log L}} \right) V
$$

## 核心要点
1. 当推理长度 $n$ 超过训练长度 $L$ 时，$\frac{\log n}{\log L} > 1$，注意力分布被"冷却"以防止过度集中
2. 无需训练，纯推理时应用
3. 常与 NTK-aware 插值和窗口注意力组合使用

## 代表工作
- [[Qwen]]: 与动态 NTK-aware 插值和 Window Attention 组合，实现 16K 上下文下 PPL 从 3168 降至 3.42
- [[Chiang & Cholak 2022]]: 首次提出注意力熵随长度增长的问题

## 相关概念
- [[NTK-aware Interpolation]]
- [[Window Attention]]
- [[Length Extrapolation]]
