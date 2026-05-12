---
type: concept
aliases: [大语言模型, Large Language Model]
---

# LLM

## 定义
基于 Transformer 架构的大规模语言模型，通过自回归语言建模预训练，展现涌现能力（上下文学习、推理等）。

## 核心要点
1. 自回归生成：$P(x_t \mid x_{<t})$
2. 隐空间（[[Residual Stream]]）编码高层概念（真值、语义）
3. 主要挑战：[[Hallucination]]（生成流畅但错误的内容）
4. 常见模型族：Llama、Mistral、Qwen 等

## 代表工作
- [[PCNet]]: 对 LLM 隐空间做密度估计检测幻觉
- GPT-4, Llama-3, Mistral, Qwen 系列

## 相关概念
- [[Residual Stream]]
- [[Hallucination]]
- [[Transformer]]
- [[Representation Engineering]]
