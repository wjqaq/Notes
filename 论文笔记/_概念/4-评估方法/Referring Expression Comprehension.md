---
type: concept
aliases: [Referring Expression Comprehension, REC, 指代表达理解]
---

# Referring Expression Comprehension

## 定义
视觉定位任务：给定图像和自然语言描述（指代表达），要求模型定位到描述所指的目标对象，输出 bounding box。

## 核心要点
1. 输入：图像 + 指代表达（如 "the person in red shirt on the left"）；输出：目标 bounding box
2. 主要基准：RefCOCO、RefCOCO+、RefCOCOg、GRIT
3. Qwen-VL 通过将坐标文本化，在通用模型中以 86.32 (RefCOCOg test) 接近 specialist SOTA

## 代表工作
- [[Qwen-VL]]: 通用模型中的顶级 REC 性能
- [[Shikra]]: 开源 REC LVLM 代表

## 相关概念
- [[Visual Grounding]]
- [[Grounded Captioning]]
- [[Bounding Box]]
