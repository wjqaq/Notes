---
type: concept
aliases: [Science Question Answering]
---

# ScienceQA

## 定义
多模态科学问答 benchmark，包含带有思维链解释的科学题目，评估模型的跨模态推理能力。

## 核心要点
1. 包含约 21k 道科学选择题，覆盖自然科学、语言科学和社会科学
2. 每题配有图像上下文和详细的思维链解释
3. 评估模型的多模态推理 + 解释生成能力
4. 在 [[Re-Align]] 中用作通用 VQA 评估（68.10 vs vanilla 66.02）

## 代表工作
- [[Re-Align]]: 通用 VQA 评估
- LLaVA: 原始评测工作

## 相关概念
- [[TextVQA]]
- [[MMBench]]
