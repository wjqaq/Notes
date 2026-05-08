---
type: concept
aliases: [Patch Embedding, 图像块嵌入]
---

# Patch Embedding

## 定义

将图像划分为固定大小的 patches，每个 patch 通过线性投影映射到 embedding 向量的技术，是 Vision Transformer 的核心组件。

## 数学形式

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \cdots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}
$$

其中 $\mathbf{x}_p^i$ 为第 $i$ 个 patch，$\mathbf{E}$ 为投影矩阵。

## 核心要点

1. 将 2D 图像转换为 1D token 序列，便于 Transformer 处理
2. 通常使用 16x16 或 14x14 的 patch 大小
3. 可从头训练或使用预训练权重初始化

## 代表工作

- [[Tuna-2]]: 完全移除视觉编码器，仅用 patch embedding 处理像素
- [[ViT]]: 首次将 patch embedding 应用于视觉任务
- [[Fuyu]]: encoder-free VLM，直接处理像素 patches

## 相关概念

- [[Vision Transformer]]
- [[Image Tokenization]]
- [[Positional Encoding]]
