---
type: concept
aliases: [clip, 对比语言图像预训练]
---

# CLIP

## 定义
OpenAI 2021 提出的图文对比学习模型，通过大规模图文对让图像与文本共享 embedding 空间。

## 核心要点
1. ViT image encoder + text encoder，用 [[Cosine Similarity]] 做对比学习
2. 零样本分类 / 检索开山之作
3. patch 特征背景偏置严重，难以直接用于稠密预测

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[LAION-400M]]
- [[Vision Transformer]]
