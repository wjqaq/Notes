---
type: concept
aliases: [Text Visual Question Answering]
---

# TextVQA

## 定义
专注于图中文字理解和阅读的视觉问答 benchmark，要求模型能识别并理解图像中的文字信息来回答问题。

## 核心要点
1. 包含约 45k 训练问题和约 5k 测试问题
2. 问题需要模型同时理解视觉场景和场景中的文字（如标志、标签等）
3. 考察模型的 OCR + 视觉理解联合能力
4. 在 [[Re-Align]] 中用作通用 VQA 评估（58.55 vs vanilla 58.18）

## 代表工作
- [[Re-Align]]: 通用 VQA 评估
- 各类 VLM 基准评测

## 相关概念
- [[ScienceQA]]
- [[MMBench]]
