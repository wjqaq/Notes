---
type: concept
aliases: [Self-Calibrated Rewards, 自校准奖励]
---

# Self-Calibrated Rewards

## 定义
TPO 中提出的自校准奖励机制，在训练过程中使用当前模型动态计算 token 级视觉锚定奖励 $c_{y_i}$，形成"奖励-优化-更强奖励"的正向反馈循环。

## 数学形式
$$
c_{y_i} = \begin{cases} a + \sigma(s_{y_i}) & \text{if } y_i \in y_w \\ a + 1 - \sigma(s_{y_i}) & \text{if } y_i \in y_l \end{cases}
$$

其中 $s_{y_i}$ 为视觉锚定分数，$a=0.5$ 为边际值，$\sigma$ 为 sigmoid 函数。

## 核心要点
1. 每个训练步用当前模型重新计算奖励，奖励随模型能力提升而更加精确
2. $c_{y_i} \in (0.5, 1.5)$，当 $s=0$ 时 $c=1$（不引入额外信号）
3. 正样本奖励与视觉锚定分数正相关，负样本负相关
4. 训练过程中正样本 $c_{y_i} \to 1.5$，负样本 $c_{y_i} \to 0.5$，收敛稳定

## 代表工作
- [[TPO]]: 提出自校准视觉锚定奖励

## 相关概念
- [[Visual-Anchored Rewards]]
- [[TPO]]
- [[Token-Level Rewards]]
