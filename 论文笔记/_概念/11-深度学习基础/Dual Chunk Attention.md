---
type: concept
aliases: [Dual Chunk Attention, DCA]
---

# Dual Chunk Attention

## 定义
一种支持训练时长度两倍的推理时长度外推的注意力机制，通过对序列进行 dual chunk 分割处理来实现高效长上下文推理。

## 核心要点
1. 训练时使用短序列，推理时可处理 2 倍长度的序列
2. 在 Qwen3 中与 YARN (4x) 组合，总外推倍数达到 4x
3. 相比直接训练长序列，计算成本更低

## 代表工作
- [[Qwen3]]: Long Context Stage 使用 DCA + YARN 实现 4 倍外推
- [[Qwen2.5]]: 前代模型也使用 DCA

## 相关概念
- [[YARN]]
- [[ABF]]
- [[RoPE]]
