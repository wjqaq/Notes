---
type: concept
aliases: [Grounded Captioning, 定位描述]
---

# Grounded Captioning

## 定义
视觉语言任务：生成图像描述的同时为描述中的名词短语提供对应的 bounding box 位置，将语言和视觉空间关联起来。

## 核心要点
1. 输入：图像；输出：包含 `<ref>短语</ref><box>坐标</box>` 的描述文本
2. 训练数据来自 GRIT、Visual Genome、RefCOCO 系列
3. Qwen-VL 阶段2将此作为7类多任务之一进行训练，数量为 8.7M 样本

## 代表工作
- [[Qwen-VL]]: 多任务训练的一部分
- [[Kosmos-2]]: 使用 GRIT 训练 grounded captioning

## 相关概念
- [[Visual Grounding]]
- [[Referring Expression Comprehension]]
- [[Bounding Box]]
