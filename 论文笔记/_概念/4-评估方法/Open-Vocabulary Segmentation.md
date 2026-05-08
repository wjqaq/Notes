---
type: concept
aliases: [开放词汇分割]
---

# Open-Vocabulary Segmentation

## 定义
在测试时可分割训练集未见过的类别（通常依赖文本 embedding）的分割任务。

## 核心要点
1. 依赖图文对齐的 patch 表征——[[CLIP]] 的 patch 特征原本质量差
2. [[LaSt-ViT]] 把 CLIP patch 直接变成可用分割特征，mIoU 翻倍以上
3. 典型基准：ADE20K、PASCAL Context、Cityscapes 零样本设置

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Semantic Segmentation]]
- [[CLIP]]
- [[ADE20K]]
