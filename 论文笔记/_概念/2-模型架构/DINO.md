---
type: concept
aliases: [dino, self-distillation with no labels]
---

# DINO

## 定义
Caron et al. 2021 提出的自监督 ViT 预训练方法，基于 teacher-student 自蒸馏。

## 核心要点
1. 无标注下学习到强 patch-level 表征，可直接用于无监督分割 / 定位
2. 学生网络预测教师的输出分布，教师是学生的 EMA
3. 是后续 [[DINOv2]] 的基础

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[DINOv2]]
- [[Vision Transformer]]
