---
type: concept
aliases: [RLHF, Reinforcement Learning from Human Feedback, 基于人类反馈的强化学习]
---

# RLHF (Reinforcement Learning from Human Feedback)

## 定义
通过人类偏好反馈训练奖励模型，再用强化学习优化语言模型输出对齐人类偏好的训练范式。

## 核心要点
1. 三阶段：监督微调 → 奖励模型训练 → PPO 强化学习
2. 在幻觉缓解中用于对齐模型输出与事实内容
3. 需大量人类标注和计算资源

## 代表工作
- Stiennon et al. (2020): 将 RLHF 应用于摘要生成
- [[MHSA]]: 与之相比无需修改 LLM 参数

## 相关概念
- [[Instruction Tuning]]
- [[Adversarial Training]]
