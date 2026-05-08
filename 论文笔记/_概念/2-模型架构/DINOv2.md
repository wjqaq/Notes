---
type: concept
aliases: [dinov2]
---

# DINOv2

## 定义
Meta 2023 发布的大规模自监督 ViT，综合 [[DINO]] + iBOT 并扩展训练数据至 LVD-142M。

## 核心要点
1. 产出的 patch 特征可直接用于分割、深度估计等稠密任务
2. 与 [[Register Tokens]] 组合可缓解 artifacts，但背景偏置仍在
3. 是 [[LaSt-ViT]] 的主要自监督基线之一

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[DINO]]
- [[Vision Transformer]]
- [[Register Tokens]]
