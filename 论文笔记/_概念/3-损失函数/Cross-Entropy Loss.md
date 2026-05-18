---
type: concept
aliases: [Cross-Entropy Loss, 交叉熵损失, CE Loss]
---

# Cross-Entropy Loss

## 定义
衡量两个概率分布之间差异的标准损失函数，在分类和生成任务中广泛使用。

## 数学形式
$$\text{CE}(p, q) = -\sum_{i} p_i \log q_i$$

## 核心要点
1. 最小化交叉熵等价于最大化似然
2. MHSA 中用于 $\mathcal{L}_{\text{LVLM}}$：约束修正后模型的输出分布与 ground-truth 一致

## 代表工作
- [[MHSA]]: $\mathcal{L}_{\text{LVLM}}$ 中使用交叉熵

## 相关概念
- [[LVLM Output Quality Loss]]
- [[Negative Log-Likelihood]]
