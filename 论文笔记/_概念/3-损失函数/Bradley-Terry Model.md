---
title: "Bradley-Terry Model"
created: 2025-05-08
tags: [loss-function, preference-learning, ranking]
---

# Bradley-Terry Model

## 定义

Bradley-Terry 模型是一种概率模型，用于预测成对比较中某一选项获胜的概率。在机器学习中，常用于偏好学习和奖励建模。

## 核心公式

$$
P(i \succ j) = \frac{\sigma_i}{\sigma_i + \sigma_j}
$$

其中 $\sigma_i, \sigma_j$ 是选项 $i, j$ 的"强度"参数。

## 在判别器训练中的应用

用于训练判别器区分监督样本 $y^+$ 和策略输出 $y^-$：

$$
\mathcal{L}_{D} = -\mathbb{E}_{(x,y^+,y^-)}[\log \sigma(D(x,y^+) - D(x,y^-))]
$$

判别器学习为监督样本分配更高分数。

## 应用场景

- [[PRISM]]: MoE 判别器训练
- [[RLHF]]: 奖励模型训练
- 偏好学习
- 排序学习

## 优势

1. **概率解释**: 提供明确的概率输出
2. **可微**: 易于优化
3. **无需绝对评分**: 仅需相对比较

## 相关概念

- [[RLHF]]: 强化学习人类反馈
- [[Preference Learning]]: 偏好学习
- [[Reward Modeling]]: 奖励建模
