---
type: concept
aliases: [Kosmos2, KOSMOS-2]
---

# Kosmos-2

## 定义
微软提出的具有 grounding 能力的大视觉语言模型，通过将 bounding box 坐标编码为特殊 token 实现空间定位。

## 核心要点
1. 将 bounding box 坐标离散化为 location tokens，融入 LLM 的 vocabulary
2. 使用 GRIT 数据集训练 grounding + captioning 能力
3. Qwen-VL 借鉴了其 grounding 数据格式（GRIT），但选择了更简单的文本化 bounding box 方案

## 代表工作
- [[Qwen-VL]]: grounding 数据格式参考
- [[Shikra]]: 另一种 grounding LVLM 方案

## 相关概念
- [[LVLM]]
- [[Visual Grounding]]
- [[GRIT]]
