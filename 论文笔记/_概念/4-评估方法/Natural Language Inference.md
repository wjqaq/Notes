---
type: concept
aliases: [NLI, 自然语言推理]
---

# Natural Language Inference (NLI)

## 定义
自然语言推理，判断一个假设句是否可以从前提句中逻辑推导出来（蕴含/矛盾/中立）的 NLP 任务。

## 核心要点
1. NLI 是事实一致性评估的语义基础：检查生成文本中的每一条"声明"是否被源文档"蕴含"
2. 经典数据集：SNLI, MultiNLI
3. Entity-Rubrics 将此逻辑迁移到视觉域：检查编辑后图像中的实体变化是否与指令预期的变换一致

## 代表工作
- Bowman et al., "A large annotated corpus for learning natural language inference", EMNLP 2015
- [[Atomic Facts Evaluation]]: 将 NLI 扩展到原子事实级验证
- [[Entity-Rubrics]]: 将 NLI 理念迁移到视觉编辑评估

## 相关概念
- [[Atomic Facts Evaluation]]
- [[Semantic Alignment]]
- [[Factuality Evaluation]]
