---
type: concept
aliases: [Image Captioning, 图像描述]
---

# Image Captioning

## 定义
视觉语言基础任务：给定图像，生成描述其内容的自然语言文本。

## 数学形式
给定图像 $I$，生成文本序列 $T = (t_1, ..., t_n)$，使 $P(T|I)$ 最大化。

## 核心要点
1. 主要基准：Nocaps（zero-shot）、Flickr30K（CIDEr 指标）、COCO Captions
2. Qwen-VL 在 Flickr30K zero-shot 上达 85.8 CIDEr，超越 Flamingo-80B
3. 在 Qwen-VL 的三阶段训练中，captioning 数据贯穿阶段1（1.4B 图文对）和阶段2（19.7M captioning 样本）

## 代表工作
- [[Qwen-VL]]: Flickr30K SOTA（通用模型）
- [[BLIP-2]]: 强 captioning 能力

## 相关概念
- [[CIDEr]]
- [[Visual Question Answering]]
- [[LVLM]]
