---
type: concept
aliases: [QK-Norm, Query-Key Normalization]
---

# QK-Norm

## 定义
在注意力计算中对 Query 和 Key 分别应用 LayerNorm/RMSNorm，防止注意力 logits 过大导致训练不稳定，尤其在大规模模型中是关键训练稳定性组件。

## 数学形式
$$Q = \text{Norm}(XW_Q), \quad K = \text{Norm}(XW_K)$$
$$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$

## 核心要点
1. Qwen3 新增引入（Qwen2 中使用 QKV-bias，无 QK-Norm）
2. 与 Qwen3 移除 QKV-bias 同时进行，QK-Norm 替代 bias 的稳定作用
3. 在 Large Vision Transformer（ViT-22B）中首次被证明有效
4. 对 MoE 大模型的训练稳定性尤其关键

## 代表工作
- [[Qwen3]]: Dense 和 MoE 模型均使用 QK-Norm

## 相关概念
- [[RMSNorm]]
- [[Grouped Query Attention]]
