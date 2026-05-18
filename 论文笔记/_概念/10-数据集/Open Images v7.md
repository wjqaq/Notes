---
type: concept
aliases: [Open Images Dataset, OpenImages v7]
---

# Open Images v7

## 定义
Google 发布的大规模开源图像数据集，包含统一图像分类、目标检测和视觉关系检测标注。v7 版本有约 9M 图像，Entity-Rubrics 论文从中选取 1300 张复杂多实体场景作为上下文图像。

## 核心要点
1. 大规模、多样性高的开源数据集
2. 在 AbstractEdit 中，优先选择含多个交互实体的复杂场景
3. 作为基准的上下文图像来源，任何其人口统计/地理偏差会传递到 AbstractEdit

## 代表工作
- Kuznetsova et al., "The Open Images Dataset v4", IJCV 2020
- [[Entity-Rubrics]]: AbstractEdit 的图像来源

## 相关概念
- [[AbstractEdit]]
- [[Entity-Rubrics]]
