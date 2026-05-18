---
type: concept
aliases: [Adjusted Base Frequency, ABF]
---

# ABF

## 定义
Adjusted Base Frequency，一种在长上下文微调阶段提高 RoPE 基础频率的技术，使模型能够泛化到训练时未见过的更长序列长度。

## 数学形式
将 RoPE 的 base frequency $\Theta$ 从 10,000 提升到 1,000,000：
$$\Theta' = \Theta \times \text{scale\_factor}$$
其中 Qwen3 在 Long Context Stage 使用 ABF 将 base 从 10k 提升至 1M。

## 核心要点
1. 适用于长上下文扩展训练阶段
2. 与 YARN 和 Dual Chunk Attention 配合实现推理时 4 倍外推
3. Qwen3 在预训练第三阶段使用

## 代表工作
- [[Qwen3]]: Long Context Stage 使用 ABF + YARN + DCA

## 相关概念
- [[RoPE]]
- [[YARN]]
- [[Dual Chunk Attention]]
