---
type: concept
aliases: [对比流形, 事实流形, Factual Manifold]
---

# Contrastive Manifold

## 定义
通过对比学习训练的隐空间密度模型所定义的几何区域：高密度区域对应事实性 token 的隐状态投影，低密度区域对应幻觉性偏离。

## 数学形式

对比流形优化损失：

$$
\mathcal{L} = \alpha \mathbb{E}_{h^{+}}[-\log\mathcal{C}_{\text{root}}(z^{+})] + (1-\alpha) \mathbb{E}_{h^{+},h^{-}}[\max(0, \gamma + \log\mathcal{C}_{\text{root}}(z^{-}) - \log\mathcal{C}_{\text{root}}(z^{+}))]
$$

## 核心要点
1. 生成项构建覆盖事实状态的高密度流形
2. 对比项强制幻觉状态被推入边界 $\gamma$ 外的低密度区
3. $\alpha \in [0,1]$ 控制生成与对比的平衡（[[PCNet]] 中 $\alpha=0.8$）
4. 提供统计一致性保证（Appendix A, Proposition 1）

## 代表工作
- [[PCNet]]: 用对比流形学习训练 PCNet 检测幻觉
- Marks & Tegmark (2024): 真值在隐空间中几何编码

## 相关概念
- [[Density Estimation]]
- [[Probabilistic Circuit]]
- [[Contrastive Learning]]
- [[Negative Log-Likelihood]]
