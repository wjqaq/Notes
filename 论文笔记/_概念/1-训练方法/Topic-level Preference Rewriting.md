---
category: 1-训练方法
aliases: [TPR, Topic-level Preference Rewriting]
---

# Topic-level Preference Rewriting (TPR)

一种用于缓解 VLM 幻觉的数据策划框架。核心思路是将 VLM 响应分解为语义单元 (semantic units)、按 topic 聚类、通过 intra-topic self-resampling 生成替代候选、再通过选择性替换构造偏好对 $(y_w, y_l)$ 用于 DPO 训练。TPR 通过 topic 级别的精细控制实现对 reward gap configuration 的系统性优化。

## 代表工作
- [[TPR]]: Topic-level Preference Rewriting (NeurIPS 2025)
