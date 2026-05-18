---
type: concept
aliases: [FFN, 前馈网络, feedforward layer]
---

# Feed-Forward Network

## 定义
Transformer 中的全连接子层，对每个位置的 token 独立进行非线性变换，是模型容量的重要来源。

## 数学形式

$$FFN(x) = W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2$$

其中 $\sigma$ 为激活函数（Qwen 系列使用 SwiGLU）。

## 核心要点
1. 在 MoE 架构中被替换为多个专家 FFN + 路由机制
2. Qwen2.5 使用 SwiGLU 激活函数
3. FFN 参数量通常占 Transformer 总参数的约 2/3

## 代表工作
- [[Qwen2.5]]: 在 Turbo/Plus 中用 MoE 层替代标准 FFN

## 相关概念
- [[Transformer]]
- [[SwiGLU]]
- [[Mixture of Experts|混合专家模型]]
