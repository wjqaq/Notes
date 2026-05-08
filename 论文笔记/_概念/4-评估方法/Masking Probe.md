---
type: concept
aliases: [渐进遮盖诊断, masking probe]
---

# Masking Probe

## 定义
按 [[Patch Score]] 排序后从高到低逐步遮盖 patch，观察分类精度下降曲线的诊断方法。

## 核心要点
1. 若高分 patch 是真实语义贡献源，遮掉它们精度应快速下降
2. [[LaSt-ViT]] 发现原始 ViT 反而是遮掉前景（低分 patch）精度掉得更快——'惰性聚合'证据
3. 无需额外标注即可评估

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Patch Score]]
- [[LaSt-ViT]]
