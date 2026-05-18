---
type: concept
aliases: [Grounding DINO 1.5]
---

# Grounding DINO

## 定义
基于 DINO 检测器的开放词汇目标检测模型，通过将文本查询与视觉特征进行跨模态融合，实现任意类别对象的检测和定位。

## 核心要点
1. 结合 DINO（Detection Transformer）与预训练语言模型实现开放词汇检测
2. Qwen2.5-VL 使用其作为数据合成工具，生成 grounding 数据标注
3. 在 ODinW-13 上 mAP=55.0，是通用模型中 grounding 能力的强基线

## 代表工作
- [[Qwen2.5-VL]]: 使用 Grounding DINO 合成定位训练数据
- [[Grounding DINO 1.5]]: 增强版本

## 相关概念
- [[DINO]]
- [[Visual Grounding]]
- [[Open-Vocabulary Detection]]
- [[Copy-Paste Augmentation]]
