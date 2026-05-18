---
type: concept
aliases: [Dense Model, Dense Transformer]
---

# Dense Model

## 定义
每个 token 都会激活所有参数的标准 Transformer 架构变体，与 Mixture of Experts 相对，没有稀疏激活机制。

## 核心要点
1. 所有参数对所有 token 都参与计算
2. 参数量与计算量成正比
3. Qwen3 系列有 6 个 Dense 模型：0.6B, 1.7B, 4B, 8B, 14B, 32B
4. Dense 模型在小规模时通常比等激活参数的 MoE 更易训练

## 代表工作
- [[Qwen3]]: 6 个 Dense 变体
- [[Qwen2.5]]: Dense + MoE 双架构

## 相关概念
- [[Mixture of Experts]]
- [[Transformer]]
