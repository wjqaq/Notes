---
type: concept
aliases: [复制粘贴增强, Copy-Paste Data Augmentation]
---

# Copy-Paste Augmentation

## 定义
一种数据增强技术，通过将目标对象从一张图像复制并粘贴到另一张图像的背景中，合成新的训练样本，常用于目标检测和实例分割。

## 核心要点
1. 通过合成重叠、遮挡等自然场景来增强训练数据多样性
2. Qwen2.5-VL 使用此技术扩充 grounding 数据集
3. 与 Grounding DINO 和 SAM 配合，构建自动化合成流水线
4. 帮助提升开放词汇检测（ODinW mAP=43.1）的表现

## 代表工作
- [[Qwen2.5-VL]]: 用于扩充定位数据集
- [[Simple Copy-Paste]]: 原始提出，证明简单复制粘贴是强数据增强方法

## 相关概念
- [[Grounding DINO]]
- [[SAM]]
- [[Visual Grounding]]
