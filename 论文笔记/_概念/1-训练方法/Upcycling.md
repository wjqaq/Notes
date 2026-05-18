---
type: concept
aliases: [稀疏上采样, Sparse Upcycling, 模型上采样]
---

# Upcycling

## 定义
一种 MoE 模型初始化方法：从预训练的密集模型权重出发，将 FFN 复制多份并引入随机扰动来初始化 MoE 专家。

## 核心要点
1. 将密集模型的 FFN 权重复制 $\lceil n \times h_E / h_{FFN} \rceil$ 次
2. 在中间维度打乱参数以保证专家多样性
3. 每个细粒度专家随机重初始化 50% 参数引入随机性
4. 相比从头训练 MoE，可显著加速 MoE 模型的训练收敛

## 代表工作
- [[Qwen2]]: 从 Qwen2-7B 密集模型 upcycle 到 Qwen2-57B-A14B MoE

## 相关概念
- [[Mixture-of-Experts]]
- [[Fine-grained Experts]]
