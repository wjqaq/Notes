---
type: concept
aliases: [Calibrated Self-Rewarding, CSR]
---

# CSR

## 定义
Calibrated Self-Rewarding Vision Language Models，一种基于 on-policy DPO 的幻觉缓解方法，通过迭代自奖励机制选择偏好对，并引入 CLIP 计算图文相关性作为额外奖励信号。

## 核心要点
1. 采用 on-policy DPO 策略，每轮迭代用当前模型自评选择偏好对
2. 引入 CLIP 模型计算生成文本与视觉信息的相关性分数作为辅助奖励
3. 需要额外模型（CLIP），降低了训练效率
4. 仅提供句子级奖励，缺乏 token 级精细调控

## 代表工作
- [[CSR|Zhou et al. 2024]]: Calibrated Self-Rewarding Vision Language Models

## 相关概念
- [[Direct Preference Optimization]]
- [[CLIP]]
- [[TPO]]
