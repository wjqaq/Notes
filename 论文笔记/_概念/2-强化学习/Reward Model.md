---
type: concept
aliases: [奖励模型, RM, reward modeling]
---

# Reward Model

## 定义
在 RLHF 流程中，用于评估语言模型输出质量的模型，输出标量奖励分数，指导策略模型的强化学习优化。

## 核心要点
1. 通常从 SFT 模型初始化，在人类偏好数据上训练
2. Qwen2.5-RM-72B 使用 6 个评估维度：真实性、帮助性、简洁性、相关性、无害性、去偏见
3. RM 基准分数不一定能预测下游 RL 模型性能（Goodhart's Law）
4. 在线 RL 中按响应评分的方差排序查询，优先处理高分差查询

## 代表工作
- [[Qwen2.5]]: Qwen2.5-RM-72B，在 PPE 和 Human-Preference-Chinese 上领先
- [[RLHF]]: RM 是 RLHF 的核心组件

## 相关概念
- [[RLHF|人类反馈强化学习]]
- [[Direct Preference Optimization|DPO]]
- [[GRPO]]
- [[Reinforcement Learning|强化学习]]
