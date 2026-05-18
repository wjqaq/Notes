---
type: concept
aliases: [自注意力, 缩放点积注意力, Self-Attention, Scaled Dot-Product Attention]
---

# Attention

## 定义
Transformer 的核心计算单元，通过 query-key-value 机制动态聚合序列中不同位置的信息，使每个 token 能够直接访问序列中所有其他 token。

## 数学形式
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

## 核心要点
1. Query 与 Key 计算相似度分数，除以 $\sqrt{d_k}$ 防止梯度过小
2. Softmax 归一化后得到注意力权重，加权聚合 Value
3. 多头注意力（Multi-Head Attention）将多个注意力并行计算后拼接
4. 因果注意力（Causal Attention）通过 mask 禁止关注未来 token

## 代表工作
- [[Qwen2-VL]]: 结合 [[MRoPE]] 的多模态注意力
- [[Vision Transformer]]: 将自注意力应用于图像 patch

## 相关概念
- [[RoPE]]
- [[MRoPE]]
- [[KV Cache]]
- [[Window Attention]]
