---
type: concept
aliases: [Preference Alignment, 偏好对齐]
---

# Preference Alignment

## 定义
通过优化模型输出使其与人类偏好保持一致的技术范式。在 LVLM 幻觉缓解中，偏好对齐直接优化模型输出以符合事实性和视觉一致性的偏好，无需修改推理时的解码策略。

## 核心要点
1. 直接优化输出偏好，从根源上改善模型行为
2. 推理时不引入额外计算开销（与解码策略方法互补）
3. 代表方法：DPO、RLHF、PPO、GRPO 等
4. 在幻觉缓解中，偏好对齐方法和解码策略方法是两个平行的重要方向，可互补使用

## 代表工作
- [[Direct Preference Optimization|DPO (Rafailov et al. 2024)]]: 直接偏好优化
- [[RLHF|RLHF (Sun et al. 2023)]]: 基于人类反馈的强化学习
- [[TPO]]: Token 级偏好优化

## 相关概念
- [[Direct Preference Optimization]]
- [[RLHF]]
- [[TPO]]
- [[PPO]]
- [[GRPO]]
