---
type: concept
aliases: [QKV偏置, attention bias, qkv bias]
---

# QKV Bias

## 定义
在 Transformer 注意力机制中，对 Query、Key、Value 投影添加可学习的偏置项 $b_Q, b_K, b_V$，有助于提升长度外推能力。

## 数学形式

$$Q = XW_Q + b_Q, \quad K = XW_K + b_K, \quad V = XW_V + b_V$$

## 核心要点
1. 相对于无偏置注意力，QKV Bias 增强了模型长度泛化能力
2. 配合 RoPE 使用时效果更好
3. Qwen2 和 Qwen2.5 系列均采用此设计

## 代表工作
- [[Qwen2.5]]: 所有 Dense 模型使用 QKV Bias
- [[Qwen2]]: 首次在 Qwen 系列中引入

## 相关概念
- [[Attention]]
- [[RoPE|旋转位置编码]]
- [[GQA|分组查询注意力]]
