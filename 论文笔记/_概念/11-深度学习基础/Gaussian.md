---
type: concept
aliases: [高斯分布, 正态分布, Normal Distribution]
---

# Gaussian

## 定义
最常见的连续概率分布，由均值 $\mu$ 和标准差 $\sigma$ 参数化，概率密度呈钟形曲线。在 [[PCNet]] 中作为异构混合叶节点的组成分布之一。

## 数学形式

$$
P(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

## 核心要点
1. 薄尾分布，对异常值敏感
2. LLM 隐空间呈现重尾几何，单一 Gaussian 不足以建模
3. [[PCNet]] 将 Gaussian 与 [[Laplace]]、[[Student-T]] 混合使用

## 相关概念
- [[Laplace]]
- [[Student-T]]
- [[Probabilistic Circuit]]
- [[Density Estimation]]
