---
type: concept
aliases: [RoPE, 旋转位置编码]
---

# Rotary Position Embedding (RoPE)

## 定义
一种通过旋转变换将相对位置信息编码到注意力机制中的位置编码方法。通过不同频率的旋转角度，使 query-key 内积自然包含相对位置信息。

## 数学形式
$$\mathbf{q}_m^\top \mathbf{k}_n = (\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$$

其中 $\mathbf{R}_m$ 是旋转角度为 $m\theta$ 的分块对角旋转矩阵。

## 核心要点
1. 自然编码相对位置，无需学习参数
2. 支持序列长度外推
3. M-RoPE 和 TM-RoPE 将 RoPE 扩展到多模态场景，分解为 temporal/height/width 维度

## 代表工作
- [[Qwen3-Omni]]: TM-RoPE 用于多模态时间对齐

## 相关概念
- [[Multimodal Rotary Position Embedding|TM-RoPE]]
