---
type: concept
aliases: [逐元素积, 哈达玛积, element-wise product]
---

# Hadamard Product

## 定义
两个同维矩阵/向量对应位置逐元素相乘，记作 $\odot$。

## 核心要点
1. 区别于矩阵乘法，只逐位置相乘
2. 在 [[LaSt-ViT]] 中用于频域掩模：$\mathbf{x}_{LP} = \mathbf{x}_{FFT} \odot \mathbf{g}$

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[1D FFT]]
- [[Low-pass Filter]]
