---
type: concept
aliases: [不共享嵌入, Untied Weights]
---

# Untied Embedding

## 定义
Transformer 模型中输入词嵌入矩阵与输出投影矩阵不共享权重的设计选择，以略微增加内存为代价换取更好的模型性能。

## 核心要点
1. 传统做法（如 GPT-2 之前）常共享输入/输出嵌入权重以减少参数
2. Untied Embedding 允许输入和输出学习不同的表示，提升下游任务性能
3. 代价是增加 $d_{\text{model}} \times |V|$ 的参数量（$|V|$ 为词表大小）

## 代表工作
- [[Qwen]]: 选择 Untied Embedding 以获取更好的性能，代价是额外内存
- [[LLaMA]] / [[LLaMA 2]]: 同样使用 untied embedding

## 相关概念
- [[Transformer]]
- [[Byte Pair Encoding]]
