---
type: concept
aliases: [SwiGLU Activation, Swish-Gated Linear Unit]
---

# SwiGLU

## 定义
一种结合 Swish 激活函数和门控线性单元（GLU）的激活函数，被广泛用于现代大语言模型和视觉Transformer中。

## 数学形式
$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$，$\sigma$ 为 sigmoid 函数，$\odot$ 为逐元素乘法。

## 核心要点
1. 相比 ReLU/GELU，在多个 NLP 和 CV 任务上表现更好
2. 引入了额外的参数矩阵 $W_2$（门控权重），但计算效率仍较高
3. Qwen2.5-VL 在 ViT 中使用 SwiGLU 以对齐 LLM 设计范式

## 代表工作
- [[Qwen2.5-VL]]: ViT 的 FFN 使用 SwiGLU 激活
- [[Qwen2.5 LLM]]: LLM 中使用 SwiGLU
- [[Qwen3-VL]]: 延续使用

## 相关概念
- [[RMSNorm]]
- [[Vision Transformer]]
- [[LLM]]
