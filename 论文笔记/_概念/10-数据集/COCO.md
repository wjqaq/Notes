---
type: concept
aliases: [ms-coco, mscoco, common objects in context]
---

# COCO

## 定义
80 类检测 / 分割 / caption 的综合基准数据集，118K 训练图。

## 核心要点
1. 检测、分割、caption、keypoint 等多任务共用
2. 开放词汇检测常把 COCO 类划分为 base + novel 评估
3. [[LaSt-ViT]] 在其 novel split 上测 [[Open-Vocabulary Detection]]

## 代表工作
- [[LaSt-ViT]]: 在其 novel split 上测 [[Open-Vocabulary Detection]]
- [[MHSA]]: POPE-COCO 和 COCO Caption 用于幻觉抑制评测
- [[POPE]]: 基于 COCO 图像构建的二分类幻觉评测

## 相关概念
- [[Open-Vocabulary Detection]]
