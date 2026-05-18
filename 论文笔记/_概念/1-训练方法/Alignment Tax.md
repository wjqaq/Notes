---
type: concept
aliases: [对齐税, Alignment Penalty]
---

# Alignment Tax

## 定义
LLM 在经历对齐训练（SFT/RLHF）后，在某些能力（如知识、推理）上的性能反而下降的现象。

## 核心要点
1. 对齐过程可能使模型在某些 benchmark 上表现退化
2. 缓解方法之一是在 PPO 训练中加入预训练数据梯度（pretrained gradient），保持原始能力
3. KL 惩罚也对缓解对齐税有一定作用

## 代表工作
- [[Qwen]]: 通过预训练梯度和足够的 KL 惩罚缓解对齐税，发现过大的预训练梯度权重会阻碍对齐效果
- [[Ouyang et al. 2022]]: InstructGPT 中也观察到对齐税现象

## 相关概念
- [[RLHF]]
- [[PPO]]
- [[Supervised Fine-Tuning]]
