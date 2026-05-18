---
type: concept
aliases: [FlashAttention, 快速注意力]
---

# Flash Attention

## 定义
一种 IO-aware 的精确注意力算法，通过分块（tiling）和重计算（recomputation）减少 GPU 高带宽内存（HBM）访问，在保持数学等价的同时大幅降低显存占用和加速训练/推理。

## 数学形式
传统注意力：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Flash Attention 在数学上等价，但在工程实现上通过 Kernel Fusion 避免将完整的注意力矩阵写入 HBM。

## 核心要点
1. 将 Q、K、V 分块加载到 SRAM，在片上计算 softmax 再写回 HBM
2. 使用 online softmax 技巧避免存储完整注意力矩阵
3. 反向传播时重计算注意力矩阵而非从 HBM 读取
4. 在长序列训练中可将显存从 $O(n^2)$ 级优化至 $O(n)$

## 代表工作
- [[Qwen]]: 在预训练和代码继续预训练中使用 Flash Attention 提升效率
- [[LLaMA]] / [[LLaMA 2]]: 采用 Flash Attention 加速训练

## 相关概念
- [[Transformer]]
- [[Attention]]
