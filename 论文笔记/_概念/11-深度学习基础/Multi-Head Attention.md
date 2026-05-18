---
type: concept
aliases: [MHA, 多头注意力]
---

# Multi-Head Attention

## 定义
Transformer 中的标准注意力机制，将输入投影到多个并行的注意力头中，每个头学习不同的注意力模式，最后拼接输出。

## 核心要点
1. $h$ 个注意力头并行计算，每个头有独立的 $W_Q, W_K, W_V$ 投影矩阵
2. 相比于单头注意力，能同时关注不同表示子空间中的信息
3. GQA 和 MQA 是其内存高效的变体

## 代表工作
- [[Qwen2]]: 采用 GQA 替代 MHA 以降低 KV cache 内存

## 相关概念
- [[Grouped Query Attention|GQA]]
- [[Attention]]
- [[Transformer]]
- [[KV Cache]]
