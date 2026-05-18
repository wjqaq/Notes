---
type: concept
aliases: [Optical Character Recognition, 光学字符识别, 文字识别]
---

# OCR

## 定义
光学字符识别技术，从图像中检测和识别文字内容，是连接视觉和文本理解的桥梁任务。

## 核心要点
1. 在 LVLM 中，OCR 能力是实现 text-oriented VQA、文档理解、图表分析等任务的基础
2. Qwen-VL 通过以下方式获得强大 OCR 能力：SynthDoG 合成数据（英文 41 种字体 + 中文 11 种字体）+ Common Crawl PDF/HTML 渲染数据，共 24.8M 样本
3. 高分辨率输入（448x448）对 OCR 至关重要——分辨率提升后可读取更小的文字

## 代表工作
- [[Qwen-VL]]: 多任务预训练中包含 24.8M OCR 样本
- [[Text-oriented VQA]]: OCR 能力的下游评估

## 相关概念
- [[Text-oriented VQA]]
- [[DocVQA]]
- [[LVLM]]
