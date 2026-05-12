---
type: concept
aliases: [信息瓶颈, IB, Information Bottleneck Principle]
---

# Information Bottleneck

## 定义
一种表征学习原则：在保留输入中与目标相关的信息的同时，压缩无关信息。形式上最大化 $I(Z;Y) - \beta I(Z;X)$，其中 $I$ 为互信息。

## 数学形式

在 [[PCNet]] 中的应用（降维投影）：

$$
z = f_{\phi}(h) \in \mathbb{R}^{D_{PC}}, \quad D_{PC} \ll D_{LLM}
$$

MLP bottleneck 将 4096 维降至 128 维，过滤语法噪声，保留语义和事实几何。

## 核心要点
1. 高维隐状态含大量语法噪声，密度估计需先降维
2. [[PCNet]] 用 2 层 MLP（ReLU）作为信息瓶颈 $f_{\phi}$
3. 瓶颈维度需权衡：太小丢信息，太大引入噪声（消融确认 $d=128$ 最优）
4. 瓶颈与 PCNet 联合端到端优化

## 代表工作
- [[PCNet]]: 用 MLP bottleneck 压缩 LLM 隐状态供 PCNet 密度估计
- Tishby et al. (2000): 信息瓶颈理论
- Variational Information Bottleneck (Alemi et al. 2017)

## 相关概念
- [[Residual Stream]]
- [[Density Estimation]]
- [[Mutual Information]]
