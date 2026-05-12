---
type: concept
aliases: [学生t分布, t分布, t-Distribution]
---

# Student-T

## 定义
一种重尾连续概率分布，由位置 $\mu$、尺度 $s$ 和自由度 $\nu$ 参数化。自由度越小尾部越重（$\nu \to \infty$ 时趋近 [[Gaussian]]）。在 [[PCNet]] 中用于捕捉 LLM 隐空间的重尾异常几何。

## 数学形式

$$
P(x \mid \mu, s, \nu) \propto \left(1 + \frac{(x-\mu)^2}{\nu s^2}\right)^{-\frac{\nu+1}{2}}
$$

## 核心要点
1. 重尾特性适合建模 LLM 隐空间的极端异常值
2. [[PCNet]] 的异构混合叶节点将 Student-T 与 [[Gaussian]]、[[Laplace]] 混合
3. 自由度 $\nu$ 在 PCNet 中与 $\mu, s$ 共享参数

## 相关概念
- [[Gaussian]]
- [[Laplace]]
- [[Probabilistic Circuit]]
- [[Density Estimation]]
- [[Anomaly Detection]]
