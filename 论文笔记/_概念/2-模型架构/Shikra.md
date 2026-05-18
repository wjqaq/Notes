---
type: concept
aliases: [Shikra]
---

# Shikra

## 定义
首个将 grounding 能力赋予多模态 LLM 的开源工作之一，通过将 bounding box 坐标表示为自然语言数字串，使模型原生支持指代对话。

## 核心要点
1. 核心洞察：bounding box 坐标可以作为普通文本 token 处理，无需特殊的位置 token
2. 支持三种 grounding 任务：referring expression comprehension、grounded captioning、referential dialogue
3. Qwen-VL 直接采用了其 bounding box 文本化思路（`<box>(x1,y1),(x2,y2)</box>`）

## 代表工作
- [[Qwen-VL]]: bounding box 格式直接参考
- [[Kosmos-2]]: 另一种 grounding tokenization 方案

## 相关概念
- [[LVLM]]
- [[Visual Grounding]]
- [[Bounding Box]]
