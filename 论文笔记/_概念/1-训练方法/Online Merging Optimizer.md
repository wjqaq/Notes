---
type: concept
aliases: [在线合并优化器, OMO]
---

# Online Merging Optimizer

## 定义
一种用于 LLM 对齐训练的优化器，在训练过程中实时合并模型参数更新，以提升奖励并减少对齐税（alignment tax）。

## 核心要点
1. 专为偏好优化（如 DPO）设计
2. 在线更新策略减少 reward hacking
3. Qwen2.5 在离线 DPO 阶段使用，学习率 $7\times10^{-7}$，1 epoch

## 代表工作
- [[Qwen2.5]]: 离线 RL (DPO) 阶段使用的优化器
- Lu et al. (2024a): 原始提出论文

## 相关概念
- [[Direct Preference Optimization|DPO]]
- [[RLHF|人类反馈强化学习]]
- [[Reinforcement Learning|强化学习]]
