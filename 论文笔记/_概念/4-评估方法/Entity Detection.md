---
type: concept
aliases: [实体检测]
---

# Entity Detection

## 定义
Entity-Rubrics 评估框架的第一阶段，VLM 被提示识别并分类上下文图像和编辑后图像中的所有实体为 Things（离散物体）、Stuff（无定形背景元素）和 Global（全局属性）。

## 核心要点
1. 三类实体：Things（物体如 "woman's face"）、Stuff（环境如 "grass"）、Global（如 "lighting"）
2. 确保评估覆盖图像的每一层，从离散物体到全局氛围
3. Stuff 分类源自 COCO-Stuff 数据集

## 代表工作
- [[Entity-Rubrics]]: 将实体检测作为三步评估的第一阶段

## 相关概念
- [[Entity Ranking]]
- [[Final Scoring]]
- [[Entity-Rubrics]]
