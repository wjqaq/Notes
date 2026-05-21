---
type: concept
aliases: [Ocean-OCR, ocean ocr benchmark]
---

# Ocean-OCR

## 定义
面向通用 OCR 应用的评测基准，涵盖文档提取、场景文本识别和手写识别三大维度，评估 VLM-based OCR 系统的综合能力。

## 核心要点
1. 三大评测维度：Document Extraction (EN/ZH)、Scene Text Recognition (EN/ZH)、Handwriting Recognition (EN/ZH)
2. 评测指标：Edit Distance、F1-Score、Precision、Recall、BLEU、METEOR
3. 涵盖学术、金融、日常等多领域文档，不同分辨率和噪声水平
4. 强调视觉感知与语言理解的整合，与现代 VLM-based OCR 目标一致

## 代表工作
- [[RTPrune]]: Ocean-OCR 上动态剪枝在某些维度超越 unpruned baseline

## 相关概念
- [[OmniDocBench]]
- [[olmOCR-Bench]]
