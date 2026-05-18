---
type: concept
aliases: [自回归语言建模, Autoregressive LM, Causal LM]
---

# Autoregressive Language Modeling

## 定义
一种语言模型训练范式，模型根据已生成的前文 token 序列逐 token 预测下一个 token，即 $P(x_t | x_{<t})$。

## 数学形式

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)
$$

## 核心要点
1. 训练目标为最大化给定前文条件下下一 token 的条件概率
2. 生成时逐 token 自回归采样，无法并行解码
3. 是 GPT 系列、LLaMA 系列和 Qwen 系列的基础训练范式
4. 与 BERT 的 Masked Language Modeling (MLM) 形成对比

## 代表工作
- [[Qwen]]: 预训练和 SFT 阶段均使用自回归语言建模目标
- [[GPT-3]]: 大规模自回归语言模型的标杆

## 相关概念
- [[Transformer]]
- [[LLM]]
