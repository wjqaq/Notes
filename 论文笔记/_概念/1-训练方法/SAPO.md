---
type: concept
aliases: [Smooth Adaptive Policy Optimization, 平滑自适应策略优化]
---

# SAPO

## 定义
一种平滑且自适应的策略梯度强化学习方法，用于 [[Qwen3-VL]] 的推理 RL 训练阶段。

## 核心要点
1. 基于策略梯度(Policy Gradient)的 RL 算法
2. 在不同文本和多模态任务中、以及不同模型尺寸和架构下均能提供一致改进
3. 用于 Qwen3-VL 的推理强化学习阶段，约 30K RL queries

## 代表工作
- [[Qwen3-VL]]: 使用 SAPO 进行推理 RL
- Gao et al. (2025): SAPO 原始论文

## 相关概念
- [[RLHF]]
- [[Chain-of-Thought]]
