---
type: concept
aliases: [拉普拉斯分布, Double Exponential]
---

# Laplace

## 定义
一种连续概率分布，具有比 [[Gaussian]] 更重的尾部，由位置参数 $\mu$ 和尺度参数 $b$ 参数化。在 [[PCNet]] 中作为异构混合叶节点的组成分布之一。

## 数学形式

$$
P(x \mid \mu, b) = \frac{1}{2b} \exp\left(-\frac{|x-\mu|}{b}\right)
$$

## 核心要点
1. 尾部比 Gaussian 重，对异常值更鲁棒
2. L1 正则化等价于 Laplace 先验下的 MAP 估计
3. [[PCNet]] 将 Laplace 与 [[Gaussian]]、[[Student-T]] 混合以捕捉 LLM 隐空间的重尾几何

## 相关概念
- [[Gaussian]]
- [[Student-T]]
- [[Probabilistic Circuit]]
- [[Density Estimation]]
