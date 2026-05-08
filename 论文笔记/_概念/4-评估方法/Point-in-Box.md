---
type: concept
aliases: [PiB, point in box]
---

# Point-in-Box

## 定义
在 ImageNet 单物体图上，[[Patch Score]] 最高的 patch 是否落在 ground-truth bbox 内的比例。

## 核心要点
1. 量化 ViT 是否把 CLS 锚定到前景——数值越高越好
2. [[LaSt-ViT]] 在三种监督范式下把 PiB 平均提升 10-25 点
3. 是 artifacts 现象的定量度量指标

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Patch Score]]
- [[LaSt-ViT]]
