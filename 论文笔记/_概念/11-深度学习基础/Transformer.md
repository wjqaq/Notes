---
type: concept
aliases: [Transformer架构, Transformer model]
---

# Transformer

## 定义
基于自注意力机制的序列到序列深度学习架构，由 Vaswani et al. (2017) 提出，是现代大语言模型的基础。

## 核心要点
1. 核心组件：多头自注意力 + 前馈网络 + 残差连接 + 层归一化
2. Decoder-only 变体（GPT 风格）是当前 LLM 的主流架构
3. 支持并行计算，训练效率高于 RNN

## 代表工作
- [[Qwen2.5]]: Decoder-only Transformer，含 GQA、SwiGLU、RoPE、QKV Bias、RMSNorm
- [[GPT-4o]]: Decoder-only Transformer

## 相关概念
- [[Attention]]
- [[Feed-Forward Network|前馈网络]]
- [[GQA|分组查询注意力]]
- [[RoPE|旋转位置编码]]
