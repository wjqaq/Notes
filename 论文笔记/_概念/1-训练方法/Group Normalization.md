---
title: "Group Normalization"
created: 2025-05-08
tags: [normalization, advantage-estimation, rl]
---

# Group Normalization

## 定义

组归一化（Group Normalization）是一种归一化技术，在强化学习中常用于优势函数估计，通过在组内样本间归一化来稳定训练。

## 核心公式

$$
A_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^N)}{\text{std}(\{r_j\}_{j=1}^N)}
$$

## 在优势估计中的作用

将同一提示生成的多个响应的奖励进行归一化，得到相对优势：

- 消除绝对奖励尺度的影响
- 提供相对比较信号
- 稳定策略梯度估计

## 应用场景

- [[PRISM]]: 对抗性在线策略蒸馏中的优势计算
- [[GRPO]]: 组相对策略优化
- 强化学习中的基线估计

## 优势

1. **尺度不变**: 不依赖绝对奖励值
2. **稳定训练**: 减少方差
3. **相对比较**: 突出组内差异

## 相关概念

- [[Advantage Function]]: 优势函数
- [[GRPO]]: 组相对策略优化
- [[Policy Gradient]]: 策略梯度
