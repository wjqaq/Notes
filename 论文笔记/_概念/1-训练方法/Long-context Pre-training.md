---
type: concept
aliases: [长上下文预训练, long context training]
---

# Long-context Pre-training

## 定义
在预训练后期阶段扩展模型的上下文窗口长度，使模型能够处理和理解超长序列。

## 核心要点
1. Qwen2.5 采用两阶段：4K -> 32K 上下文长度
2. Turbo 渐进式扩展：32K -> 65K -> 131K -> 262K，使用 RoPE 基础频率 10M
3. 每阶段 40% 序列为最长长度，60% 为较短序列，保持短序列性能
4. 配合 YARN + DCA 可在推理时进一步扩展到训练长度的 4 倍

## 代表工作
- [[Qwen2.5]]: Turbo 通过渐进式长上下文预训练 + YARN/DCA 达到 1M token
- [[Qwen2]]: 首次引入长上下文预训练策略

## 相关概念
- [[YARN|YaRN]]
- [[Dual Chunk Attention|DCA]]
- [[ABF]]
- [[RoPE|旋转位置编码]]
