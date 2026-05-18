---
category: 深度学习基础
tags: [metric, perceptual-similarity, image-quality]
related_works: ["Entity-Rubrics"]
created: 2026-05-18
---

# LPIPS（Learned Perceptual Image Patch Similarity）

LPIPS 是一种基于深度特征的感知图像相似度度量，由 Zhang et al. 提出。

## 计算方式

使用预训练深度网络（如 AlexNet、VGG、SqueezeNet）提取图像块特征，计算特征空间中的距离。

## 在 Entity-Rubrics 中的使用

Entity-Rubrics 论文将 LPIPS 作为上下文图像保存性的辅助度量指标。但分析发现，看似优秀的 LPIPS 分数（低距离）可能是严重编辑不足（Under-editing）的产物——模型未修改图像所以"完美保存"了原始内容。
