---
type: concept
aliases: [LRP, 逐层相关性传播, relevance propagation]
---

# Layer-wise Relevance Propagation

## 定义
一种将神经网络输出逐层反向分解为输入特征可加性贡献的解释性方法，满足 relevance 守恒律。

## 数学形式

基本传播规则（LRP-z）：

$$
\Phi_j^\ell = \sum_i \frac{a_j^\ell W_{ji}^\ell}{\sum_k a_k^\ell W_{ki}^\ell} \cdot \Phi_i^{\ell+1}
$$

守恒律：

$$
\sum_i \Phi_i^{\ell-1} = \sum_j \Phi_j^\ell = \Phi^\ell
$$

## 核心要点
1. 将模型预测 $f(x)$ 分解为 $\sum_i \Phi_i$，$\Phi_i$ 表示输入特征 $i$ 对输出的贡献
2. 通过泰勒分解的深度推广，在各层间按激活比例传播 relevance
3. LRP-$\varepsilon$ 变体在分母加小常数防止数值不稳定
4. AttnLRP 是适配 Transformer 注意力机制的 LRP 变体
5. 广泛用于模型可解释性、幻觉诊断、模型偏差分析

## 代表工作
- [[LIME]]: 首次将 LRP 用于多模态幻觉诊断与推理时缓解
- (Bach et al., 2015): LRP 原始提出

## 相关概念
- [[AttnLRP]]
- [[LRP-ε]]
