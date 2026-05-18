---
type: concept
aliases: [CIDEr, Consensus-based Image Description Evaluation]
---

# CIDEr

## 定义
图像描述质量评估指标，通过计算生成 caption 与人工参考 caption 之间的 n-gram 共识度来评分。

## 核心要点
1. 基于 TF-IDF 加权的 n-gram 匹配，对常见词（如 "a", "the"）降权
2. 主要用于 Nocaps 和 Flickr30K 的 captioning 评估
3. Qwen-VL 在 Flickr30K 上获 85.8 CIDEr（zero-shot），验证了其强大的描述能力

## 代表工作
- [[Qwen-VL]]: Flickr30K CIDEr 超越同期通用模型
- [[Image Captioning]]

## 相关概念
- [[Image Captioning]]
- [[Nocaps]]
- [[Flickr30k]]
