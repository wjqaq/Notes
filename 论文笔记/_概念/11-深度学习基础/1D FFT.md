---
type: concept
aliases: [一维快速傅里叶变换, 1D Fast Fourier Transform]
---

# 1D FFT

## 定义
将一维离散信号从时/空域转换到频域的快速算法，复杂度 $O(N \log N)$。

## 核心要点
1. 把长度 $N$ 的序列表示为各频率正弦波叠加
2. [[LaSt-ViT]] 中对 patch 的通道维做 1D FFT 以分析语义稳定性
3. 配合 [[Low-pass Filter]] 后做 [[Inverse FFT]] 得到低频重建

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Inverse FFT]]
- [[Low-pass Filter]]
- [[Frequency-aware Selective Aggregation]]
