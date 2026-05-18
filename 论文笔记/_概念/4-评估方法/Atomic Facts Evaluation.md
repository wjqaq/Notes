---
type: concept
aliases: [原子事实评估, Atomic Facts, FactScore]
---

# Atomic Facts Evaluation

## 定义
NLP 领域中一种事实一致性评估范式：将复杂生成文本分解为独立的"原子事实"，逐一验证每个原子事实与源文档的一致性。

## 核心要点
1. 核心思想：整体语义对齐可被近似为"零件之和"的验证
2. 代表方法：FactScore, TrueTeacher, TRUE, FActool
3. Entity-Rubrics 将此范式迁移到视觉域：以图像实体替代文本原子事实作为评估原子单元
4. 优势：提供可解释的细粒度诊断而非黑盒评分

## 代表工作
- FactScore (Min et al., EMNLP 2023): 长文本生成的细粒度事实精确度评估
- TrueTeacher (Gekhman et al., 2023): 用 LLM 学习事实一致性评估
- [[Entity-Rubrics]]: 将此范式迁移到视觉图像编辑评估

## 相关概念
- [[Natural Language Inference]]
- [[Entity-Rubrics]]
- [[Factuality Evaluation]]
