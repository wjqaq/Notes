---
type: concept
aliases: [频域感知选择性聚合, FFT stability score]
---

# Frequency-aware Selective Aggregation

## 定义
[[LaSt-ViT]] 的核心算子：沿通道维做 [[1D FFT]] 低通滤波后得到稳定性打分，用于挑选前景 patch。

## 核心要点
1. 假设前景 patch 的通道响应去除低频后仍有显著信息，背景 patch 则接近低频本身
2. 通过 $\mathbf{S}_{i,j}$ 打分后配合 [[Channel-wise Top-K Pooling]]
3. 不同通道独立挑选，避免强通道主导

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[LaSt-ViT]]
- [[1D FFT]]
- [[Low-pass Filter]]
- [[Channel-wise Top-K Pooling]]
