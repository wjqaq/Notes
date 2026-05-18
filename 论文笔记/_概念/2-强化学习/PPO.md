---
type: concept
aliases: [Proximal Policy Optimization, 近端策略优化]
---

# PPO

## 定义
一种基于策略梯度的强化学习算法，通过 clipped surrogate objective 限制新旧策略之间的更新幅度，实现稳定且高效的策略优化。

## 数学形式

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

其中

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
$$

## 核心要点
1. Clipping 机制防止策略更新幅度过大导致训练不稳定
2. 相比 TRPO 实现更简单，不需要二阶优化
3. 在 LLM 对齐中，PPO 通常与 KL 惩罚项和 reward model 配合使用

## 代表工作
- [[Qwen]]: 在 RLHF 阶段使用 PPO 优化 chat 模型，KL 系数 0.04
- [[LLaMA 2]]: 使用 PPO 进行 RLHF 对齐

## 相关概念
- [[RLHF]]
- [[Reward Model]]
- [[KL Divergence]]
