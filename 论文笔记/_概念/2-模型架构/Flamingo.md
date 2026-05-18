---
type: concept
aliases: [Flamingo]
---

# Flamingo

## 定义
DeepMind 提出的大视觉语言模型，以强大的 few-shot learning 能力著称，通过在 LLM 层间插入 Perceiver Resampler 融合视觉信息。

## 核心要点
1. 在 LLM 的多个层中插入 gated cross-attention 模块，将视觉特征注入语言表示
2. 使用 Perceiver Resampler 将变长视觉特征压缩为固定数量的视觉 token
3. 训练时使用大规模交错图文数据，展现强大的上下文学习能力
4. Qwen-VL 在 few-shot learning 实验中将其作为主要对比基线

## 代表工作
- [[Qwen-VL]]: few-shot 能力对比
- [[OpenFlamingo]]: 开源复现

## 相关概念
- [[LVLM]]
- [[Few-shot Learning]]
- [[Perceiver Resampler]]
