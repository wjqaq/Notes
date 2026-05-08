---
title: "Distribution Alignment"
created: 2025-05-08
tags: [alignment, distribution, training]
---

# Distribution Alignment

## 定义

分布对齐（Distribution Alignment）是一种训练技术，旨在使模型策略分布与目标分布（如监督分布或原始能力分布）保持一致。

## 问题背景

在监督微调（SFT）后，模型可能偏离：
- **原始能力分布**: 模型原有的有利分布
- **监督分布**: 演示数据的分布

这种分布漂移会在后续强化学习中复合错误。

## 解决方法

1. **显式对齐阶段**: 在 SFT 和 RL 之间插入对齐阶段
2. **对抗性训练**: 通过判别器引导策略分布
3. **在线策略蒸馏**: 在策略自身分布上学习

## 在 PRISM 中的应用

$$
\min_\theta \max_\phi \mathbb{E}[r_\phi(x,y^+) - r_\phi(x,y^-)]
$$

策略通过对抗游戏逼近监督分布。

## 评估指标

- **结构性代理**: 推理步数、描述项数
- **判别器奖励**: 监督样本与策略输出的差距
- **下游任务性能**: RLVR 阶段的表现

## 应用场景

- [[PRISM]]: 三阶段流水线的核心
- 多模态模型训练
- 大语言模型对齐

## 相关概念

- [[On-Policy Distillation]]: 实现方法
- [[PRISM]]: 应用实例
- [[SFT]]: 监督微调
