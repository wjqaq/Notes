---
type: concept
aliases: [Sinkhorn迭代, Sinkhorn distances, Sinkhorn-Knopp algorithm]
---

# Sinkhorn Algorithm

## 定义
求解熵正则化最优传输问题的迭代算法，通过对矩阵交替进行行列归一化快速逼近最优传输计划。

## 数学形式
在 log-space 迭代更新对偶变量 $\mathbf{u}, \mathbf{v}$：

$$\begin{aligned}
\mathbf{u}^{(t+1)} &= \log \boldsymbol{\mu} - \text{LogSumExp}(\mathbf{Z} + \mathbf{v}^{(t)\top}) \\
\mathbf{v}^{(t+1)} &= \log \boldsymbol{\nu} - \text{LogSumExp}(\mathbf{Z}^{\top} + \mathbf{u}^{(t+1)\top})
\end{aligned}$$

最终传输计划：$\mathbf{P} = \exp(\mathbf{Z} + \mathbf{u}\mathbf{1}^{\top} + \mathbf{1}\mathbf{v}^{\top})$

## 核心要点
1. 时间复杂度远低于原始线性规划求解，适合 GPU 并行化
2. 熵正则化参数控制解与最优传输的近似程度
3. Log-space 实现提供数值稳定性
4. 典型迭代次数 T=100 即可收敛
5. 在 RTPrune 中用于匹配 kept token 与 prop token

## 代表工作
- [[RTPrune]]: 用 Sinkhorn 算法求解 token 匹配问题（T=100, log-space）
- Cuturi (2013): Sinkhorn distances: Lightspeed computation of optimal transport

## 相关概念
- [[Optimal Transport]]
- [[Token Pruning]]
