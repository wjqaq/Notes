---
type: concept
aliases: [最优传输, OT, optimal transport theory, Kantorovich problem]
---

# Optimal Transport

## 定义
研究如何以最小代价将一个概率分布"传输"到另一个分布的数学理论，在机器学习中常用于特征匹配、分布对齐和 token 合并。

## 数学形式
给定两个分布 $\mu, \nu$ 和代价矩阵 $\mathbf{C}$，寻找最优传输计划 $\mathbf{P}$：

$$\min_{\mathbf{P}} \sum_{i,j} \mathbf{C}_{i,j} \mathbf{P}_{i,j} \quad \text{s.t.} \quad \mathbf{P}\mathbf{1} = \mu, \; \mathbf{P}^{\text{T}}\mathbf{1} = \nu$$

在 token 剪枝中转为最大化相似度：

$$\max_{\mathbf{P}} \sum_{i,j} \mathbf{S}_{i,j} \cdot \mathbf{P}_{i,j} \quad \text{s.t.} \quad \mathbf{P}\mathbf{1}_{N-M} = \mathbf{1}_M, \; \mathbf{P}^{\text{T}}\mathbf{1}_M = \mathbf{1}_{N-M}$$

## 核心要点
1. 相比贪心匹配，最优传输提供全局最优的 token 对应关系
2. 引入 dustbin 机制允许部分 token 不与任何目标匹配（被丢弃）
3. [[Sinkhorn Algorithm]] 提供快速近似求解（熵正则化 + 迭代行列归一化）
4. 在 [[SuperGlue]] 中首次用于特征匹配，RTPrune 将其推广到 token 合并

## 代表工作
- [[RTPrune]]: 用最优传输匹配 kept token 与 prop token，实现信息保留的 token 合并
- [[SuperGlue]]: 用最优传输 + GNN 进行图像特征匹配

## 相关概念
- [[Sinkhorn Algorithm]]
- [[Token Pruning]]
