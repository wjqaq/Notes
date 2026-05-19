---
category: 1-训练方法
aliases: [LLaVA-RLHF]
---

# LLaVA-RLHF

基于人工反馈的 LLaVA 对齐方法。首先在人工标注的指令微调数据集上微调 LLava，然后训练 reward model 在 10k 人类反馈偏好数据上，最后使用 PPO 在 72k 事实增强数据上进行偏好学习。

## 代表工作
- [[LLaVA-RLHF]]: Aligning Large Multimodal Models with Factually Augmented RLHF (ACL 2024)
- [[TPR]]: 在多个幻觉基准上大幅超越
