---
type: concept
aliases: [Root Mean Square Layer Normalization, RMS Layer Normalization]
---

# RMSNorm

## 定义
一种仅使用均方根（RMS）统计量进行归一化的技术，相比传统的 Layer Normalization 去除了均值中心化操作，计算更高效。

## 数学形式
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

其中 $d$ 为特征维度，$\gamma$ 为可学习的缩放参数，$\epsilon$ 为数值稳定性小量。

## 核心要点
1. 相比 LayerNorm 省略了均值减法，减少了 $O(n)$ 的计算量
2. 在 LLM 中被广泛采用（LLaMA 系列），Qwen2.5-VL 将其引入 ViT 以统一架构范式
3. 与 SwiGLU 搭配使用，是 Qwen2.5-VL ViT 的标准化选择

## 代表工作
- [[Qwen2.5-VL]]: ViT 和 LLM 均使用 RMSNorm
- [[Qwen2.5 LLM]]: 基础组件
- [[Qwen3-VL]]: 延续使用

## 相关概念
- [[SwiGLU]]
- [[Group Normalization]]
- [[Vision Transformer]]
