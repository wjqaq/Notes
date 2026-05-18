---
type: concept
aliases: [Cross Attention, 交叉注意力]
---

# Cross-Attention

## 定义
一种注意力机制，Query 来自一个序列，Key 和 Value 来自另一个序列，用于不同模态/序列之间的信息融合。

## 数学形式
$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q$ 来自目标序列，$K, V$ 来自源序列。

## 核心要点
1. Qwen-VL 的 VL Adapter 使用单层 Cross-Attention：可学习 query embeddings 作为 Q，ViT 输出作为 K/V
2. 相比 Self-Attention，Cross-Attention 天然适合模态间信息压缩和融合
3. 压缩比由 query 数量控制，Qwen-VL 使用 256 个查询将任意长度视觉特征压缩为固定长度

## 代表工作
- [[Qwen-VL]]: VL Adapter 核心机制
- [[BLIP-2]]: Q-Former 中使用 Cross-Attention

## 相关概念
- [[Attention]]
- [[Vision Transformer]]
