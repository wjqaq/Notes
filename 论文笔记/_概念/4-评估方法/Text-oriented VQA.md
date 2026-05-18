---
type: concept
aliases: [TextVQA, text-oriented VQA, 文本导向视觉问答]
---

# Text-oriented VQA

## 定义
一类特殊的视觉问答任务，要求模型阅读和理解图像中的文字才能正确回答，是衡量 LVLM 细粒度感知能力的重要维度。

## 核心要点
1. 包含子任务：TextVQA（自然场景文字）、DocVQA（文档）、ChartQA（图表）、OCR-VQA（书籍封面）、AI2D（科学图表）
2. 需要模型同时具备视觉识别、OCR 和语言理解能力
3. Qwen-VL 通过高分辨率训练（448x448）和丰富 OCR 数据在此类任务上表现优异

## 代表工作
- [[Qwen-VL]]: ChartQA (65.7)、OCR-VQA (75.7) 领先同期模型
- [[mPLUG-DocOwl]]: 专注文档理解的 LVLM

## 相关概念
- [[OCR]]
- [[Visual Question Answering]]
- [[DocVQA]]
