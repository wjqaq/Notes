---
type: concept
aliases: [Token-Level Rewards, Token级奖励, token-wise rewards]
---

# Token-Level Rewards

## 定义
对生成序列中每一个 token 分别赋予奖励信号的机制，区别于传统的对整句赋予单一奖励。token 级奖励能更精细地调整模型中幻觉高发的特定 token。

## 核心要点
1. 句子级奖励无法区分幻觉 token 和正确 token，token 级奖励可精确调控
2. 现有 token 级方法（RLHF-V、V-DPO）依赖人工标注或合成数据
3. TPO 提出无需标注的自校准 token 级奖励计算方式
4. token 级奖励可融入 DPO、PPO、GRPO 等多种 RLHF 框架

## 代表工作
- [[TPO]]: 自校准 token 级视觉锚定奖励
- [[RLHF-V]]: 基于人工细粒度标注的 token 级矫正
- [[V-DPO]]: 基于合成数据的 token 级优化

## 相关概念
- [[Visual-Anchored Rewards]]
- [[TPO]]
- [[Direct Preference Optimization]]
