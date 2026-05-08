---
title: "Minimax Objective"
created: 2025-05-08
tags: [adversarial, optimization, game-theory]
---

# Minimax Objective

## 定义

极小极大目标（Minimax Objective）是博弈论和对抗训练中的核心优化目标，一方试图最大化目标函数，另一方试图最小化。

## 核心公式

$$
\min_\theta \max_\phi \mathbb{E}[f_\phi(x) - f_\phi(g_\theta(x))]
$$

## 在对抗训练中的应用

- **判别器 $\phi$**: 最大化区分真实样本和生成样本的能力
- **生成器 $\theta$**: 最小化判别器的区分能力

在 PRISM 中：

$$
\min_\theta \max_\phi \mathbb{E}[r_\phi(x,y^+) - r_\phi(x,y^-)]
$$

判别器学习区分监督样本和策略输出，策略学习欺骗判别器。

## 应用场景

- [[PRISM]]: 对抗性在线策略蒸馏
- [[GAN]]: 生成对抗网络
- 对抗性训练
- 鲁棒优化

## 训练策略

1. **交替训练**: 判别器和策略交替更新
2. **梯度下降/上升**: 判别器梯度上升，策略梯度下降
3. **平衡**: 保持判别器和策略能力平衡

## 相关概念

- [[Adversarial Training]]: 对抗性训练
- [[GAN]]: 生成对抗网络
- [[Game Theory]]: 博弈论
