---
type: concept
aliases: [可微 TopK, Gumbel TopK]
---

# Gumbel Top-K

## 定义
通过 Gumbel 扰动与 soft relaxation 使 TopK 选择过程可微。

## 核心要点
1. 训练时将离散选择变为连续分布采样
2. 常用于稀疏注意力 / 可微结构学习
3. [[LaSt-ViT]] 讨论作为未来改进方向——让硬 TopK 可端到端训练

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Top-K Selection]]
- [[Channel-wise Top-K Pooling]]
