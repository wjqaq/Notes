---
type: concept
aliases: [概率电路, PC, Probabilistic Circuits, Sum-Product Network, SPN]
---

# Probabilistic Circuit

## 定义
一种通过有向无环图（DAG）表示联合概率分布的可处理概率模型，保证精确边缘推断和条件推断在节点数线性时间内完成。

## 数学形式

三种节点类型：

$$
\begin{aligned}
\text{Input (leaf)}: \quad \mathcal{C}_{n}(z_{\mathrm{sc}(n)}) &= q_{n}(z_{\mathrm{sc}(n)};\eta_{n}) \\
\text{Sum}: \quad \mathcal{C}_{n}(z_{\mathrm{sc}(n)}) &= \sum_{c\in\mathrm{ch}(n)} w_{n,c}\,\mathcal{C}_{c}(z_{\mathrm{sc}(c)}) \\
\text{Product}: \quad \mathcal{C}_{n}(z_{\mathrm{sc}(n)}) &= \prod_{c\in\mathrm{ch}(n)}\mathcal{C}_{c}(z_{\mathrm{sc}(c)})
\end{aligned}
$$

## 核心要点
1. 保证 smoothness 和 decomposability 以实现可处理推断
2. 支持精确 MPE/MAP 推断、边缘化、条件推断
3. 计算复杂度为节点数 $|\mathcal{N}|$ 的线性
4. 可扩展至在线结构学习、无损压缩、神经网络集成

## 代表工作
- [[PCNet]]: 使用 PC 作为 LLM 隐空间密度估计器，检测幻觉
- Poon & Domingos (2011): Sum-Product Networks 基础框架
- Choi et al. (2020): Probabilistic Circuits 统一框架

## 相关概念
- [[Density Estimation]]
- [[Negative Log-Likelihood]]
- [[Bayesian Network]]
