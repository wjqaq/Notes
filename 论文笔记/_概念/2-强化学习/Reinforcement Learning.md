---
type: concept
aliases: [RL, 强化学习, reinforcement learning]
---

# Reinforcement Learning

## 定义
机器学习范式之一，智能体通过与环境交互获取奖励信号来学习最优策略。在 LLM 对齐中，RL 用于根据人类偏好优化模型输出。

## 核心要点
1. LLM 中的 RL 主要形式：RLHF、DPO、GRPO
2. Qwen2.5 采用两阶段 RL：离线 DPO + 在线 GRPO
3. 离线 RL 处理难评估能力（推理、真实性），在线 RL 用 RM 评分优化

## 代表工作
- [[Qwen2.5]]: 两阶段 RL 对齐策略
- [[RLHF]]: 基于人类反馈的强化学习
- [[GRPO]]: 组相对策略优化

## 相关概念
- [[RLHF|人类反馈强化学习]]
- [[Direct Preference Optimization|DPO]]
- [[GRPO]]
- [[Reward Model|奖励模型]]
