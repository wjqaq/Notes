---
type: concept
aliases: [粗粒度语义监督]
---

# Coarse-grained Semantic Supervision

## 定义
仅用图像级标签（分类 label / image-level caption）训练视觉模型，不提供 patch / pixel 级监督。

## 核心要点
1. 是 [[LaSt-ViT]] 定位 artifacts 的两个根因之一
2. 由于缺少空间 grounding，模型倾向于用大量背景 patch 作为分类的 shortcut——形成'惰性聚合'
3. 相对物 = 稠密监督（分割 mask、patch contrastive 等）

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[LaSt-ViT]]
- [[Vision Transformer]]
- [[CLIP]]
