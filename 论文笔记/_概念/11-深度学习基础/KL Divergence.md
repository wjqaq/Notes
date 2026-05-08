---
type: concept
aliases: [KL散度, Kullback-Leibler divergence, KL regularization, 相对熵]
---

# KL Divergence

## 定义
衡量两个概率分布之间差异的非对称度量，在深度学习中常用作正则化项约束模型输出分布。

## 数学形式

$$
D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

## 核心要点
1. 非对称：$D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$
2. 非负：当且仅当 $P=Q$ 时取零
3. 在 [[LIME]] 中用于约束扰动后输出分布不偏离原始冻结模型太远
4. 可以加权组合：$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda D_{\text{KL}}(p_\theta \parallel p_{\text{ref}})$

## 代表工作
- [[LIME]]: KL 正则项防止推理时优化偏离原始行为

## 相关概念
- [[Inference-time Optimization]]
- [[Modality Relevance]]
