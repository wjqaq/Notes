---
type: concept
aliases: [Minference, MInference, 动态稀疏注意力]
---

# Minference

## 定义
一种动态稀疏注意力机制，用于加速长上下文 LLM 的预填充（pre-filling）阶段，通过识别并跳过不重要的注意力计算来减少计算量。

## 核心要点
1. Qwen2.5-Turbo 在 1M token 时注意力计算量减少 12.5 倍
2. TTFT 加速 3.2-4.3 倍（Turbo），2.3-5.6 倍（7B）
3. 配合 YARN + DCA 实现高效长上下文推理

## 代表工作
- [[Qwen2.5]]: 用于 Turbo 的超长上下文推理加速
- Jiang et al. (2024b): 原始提出论文

## 相关概念
- [[Dual Chunk Attention|DCA]]
- [[YARN|YaRN]]
- [[KV Cache]]
