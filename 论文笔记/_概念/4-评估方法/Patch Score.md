---
type: concept
aliases: [patch score, patch-CLS similarity]
---

# Patch Score

## 定义
每个 patch token 与 [[CLS Token]] 的 [[Cosine Similarity]]，用于度量该 patch 对全局表征的贡献。

## 核心要点
1. $\mathcal{S}_p = \frac{\mathbf{x}_{patch} \cdot \mathcal{Q}_{CLS}}{\|\mathbf{x}_{patch}\|_2 \|\mathcal{Q}_{CLS}\|_2}$
2. [[LaSt-ViT]] 用其诊断 ViT 的背景偏置——原始 ViT 的高分 patch 多在背景
3. 可直接用于零样本前景粗分割（Table 6）

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[CLS Token]]
- [[Cosine Similarity]]
- [[Point-in-Box]]
- [[Masking Probe]]
