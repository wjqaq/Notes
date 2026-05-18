---
type: concept
aliases: [Group Relative Policy Optimization, Group Policy Optimization]
---

# GRPO

## 定义
一种无需 critic 模型的策略优化算法，对每个 query 采样多个 rollout，利用组内相对优势进行策略更新，被广泛用于 LLM 推理能力强化学习。

## 数学形式
$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon) A_i \right) \right]$$

其中 $G$ 为每个 query 的 rollout 数，$A_i$ 为组内标准化后的相对优势。

## 核心要点
1. 无需额外训练 critic 模型，减少内存和计算开销
2. 组内相对优势使训练更稳定，适合 LLM 推理任务的 RL
3. 是 DeepSeek-R1 和 Qwen3 推理训练的核心算法

## 代表工作
- [[Qwen3]]: Reasoning RL 阶段使用 GRPO（3,995 query-verifier pairs）
- [[DeepSeek-R1]]: 使用 GRPO 提升推理能力

## 相关概念
- [[PPO]]
- [[Reinforcement Learning]]
- [[Thinking Mode Fusion]]
