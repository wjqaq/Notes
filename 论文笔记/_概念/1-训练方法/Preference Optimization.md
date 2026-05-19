---
type: concept
aliases: [偏好优化, Preference Learning]
---

# Preference Optimization (偏好优化)

## 定义
利用人类或 AI 的偏好信号（chosen vs rejected 响应对）来对齐模型行为的一类训练范式。

## 数学形式
通用形式为最大化 chosen 响应相对 rejected 响应的偏好概率：
$$\max_\theta \mathbb{E}_{(x, y_w, y_l)} \left[ \log P(y_w \succ y_l | x) \right]$$

## 核心要点
1. 两大类方法:
   - 基于 RL: 训练 Reward Model $\to$ 用 [[PPO]] 优化（如 [[RLHF]]）
   - 直接偏好学习: 直接在偏好数据上优化（如 [[Direct Preference Optimization|DPO]]、KTO）
2. DPO 隐式地将策略本身作为 reward model，无需单独训练 reward model
3. 偏好数据质量是决定对齐效果的关键因素
4. [[Re-Align]] 提出 rDPO，在标准 DPO 上增加视觉偏好优化项

## 代表工作
- [[Direct Preference Optimization|DPO]]: 直接偏好优化
- [[RLHF]]: 基于 PPO 的强化学习人类反馈对齐
- [[Re-Align]]: rDPO 联合文本和视觉偏好优化
- KTO: Prospect theoretic 对齐

## 相关概念
- [[Direct Preference Optimization]]
- [[RLHF]]
- [[Preference Dataset]]
- [[Hallucination]]
