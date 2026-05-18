---
type: concept
aliases: [因果注意力, Causal Mask]
---

# Causal Attention

## 定义
Transformer 解码器中使用的注意力机制变体，通过掩码矩阵确保每个 token 只能关注自身及之前的 token，保证自回归生成的因果性。

## 核心要点
1. 每个位置 $i$ 只能 attend 到位置 $j \leq i$，使用上三角掩码矩阵
2. 是 GPT 类自回归语言模型的标准注意力形式
3. 与双向注意力（如 BERT encoder）相对

## 代表工作
- [[Qwen2]]: 基于因果自注意力的 Transformer 架构

## 相关概念
- [[Attention]]
- [[Transformer]]
- [[KV Cache]]
