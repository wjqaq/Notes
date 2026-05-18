---
type: concept
aliases: [迭代思维, 推理时思考, Test-Time Thinking]
---

# Iterative Thinking

## 定义
推理时让模型进行迭代式思考/推理的机制，在图像编辑中指模型在生成编辑结果前先进行多步推理（如分解指令、规划步骤、自我反思），以更好地理解抽象意图。

## 核心要点
1. 在 Entity-Rubrics 论文中，Thinking 机制显著提升抽象指令遵循：Step1X-Think 比基础版 +5.3%，Bagel-Think 比基础版 +30.3%
2. 副作用：Thinking 模型在显式指令上存在"精确性税"（过度思考损害直接执行）
3. Thinking 机制是闭源模型优势的关键驱动力之一

## 代表工作
- Step1X-Think-Reflect: 开源 Thinking 图像编辑模型
- Gemini 3.x: 闭源模型内置推理时思考
- [[Entity-Rubrics]]: 首次量化 Thinking 在抽象编辑中的增益

## 相关概念
- [[Chain-of-Thought]]
- [[Entity-Rubrics]]
- [[VLM Judge]]
