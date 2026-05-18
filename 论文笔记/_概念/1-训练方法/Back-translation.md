---
type: concept
aliases: [反向翻译, back translation, reverse translation]
---

# Back-translation

## 定义
一种数据增强技术，利用翻译模型将目标语言文本翻译回源语言，生成平行训练数据。在 LLM 后训练中，用于从预训练语料生成长文本问答对。

## 核心要点
1. Qwen2.5 使用反向翻译从预训练语料生成长文本查询
2. 配合输出长度约束和 Qwen2 质量过滤
3. 有效提升长文本生成能力（从 2K 到 8K tokens）

## 代表工作
- [[Qwen2.5]]: 通过反向翻译构建长响应 SFT 数据集

## 相关概念
- [[Supervised Fine-Tuning|监督微调]]
- [[Long-context Pre-training|长上下文预训练]]
