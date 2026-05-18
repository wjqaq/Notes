---
type: concept
aliases: [Segment Anything Model, SAM2]
---

# SAM

## 定义
Meta 发布的通用图像分割基础模型，能够根据点、框或文本提示对任意图像中的任意对象进行分割，被广泛用作数据标注和合成工具。

## 核心要点
1. 基于提示的分割：接受点/框/掩码作为提示输出分割掩码
2. Qwen2.5-VL 使用 SAM 作为数据合成流水线的一部分，生成精确的对象标注
3. 与 Grounding DINO 配合，构建自动化的 grounding 训练数据

## 代表工作
- [[Qwen2.5-VL]]: 使用 SAM 合成 grounding 训练数据
- [[Qwen3-VL]]: 延续使用

## 相关概念
- [[Grounding DINO]]
- [[Visual Grounding]]
- [[Copy-Paste Augmentation]]
