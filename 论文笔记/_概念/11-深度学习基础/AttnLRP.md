---
type: concept
aliases: [Attention LRP, Attention-aware LRP]
---

# AttnLRP

## 定义
将 [[Layer-wise Relevance Propagation|LRP]] 适配到 Transformer 注意力机制的专用传播规则集合，由 Achtibat et al. (2024) 提出。

## 数学形式

注意力-值交互传播:

$$
\Phi_{ji}^{\ell-1} = \sum_p \frac{A_{ji}^\ell V_{ip}^\ell}{2 O_{jp}^\ell + \varepsilon} \cdot \Phi_{jp}^\ell
$$

## 核心要点
1. 包含针对 Linear、Softmax、Attention-Value 交互、LayerNorm/RMSNorm 的专用规则
2. 保留注意力矩阵的成对贡献结构
3. 保证跨层 relevance 守恒
4. 在 [[LIME]] 中被用作核心归因工具，量化每个 token 对输出的贡献度

## 代表工作
- [[LIME]]: 用 AttnLRP 计算 token relevance 以指导推理时优化
- (Achtibat et al., 2024): AttnLRP 原始提出

## 相关概念
- [[Layer-wise Relevance Propagation|LRP]]
