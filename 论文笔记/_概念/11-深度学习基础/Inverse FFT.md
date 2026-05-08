---
type: concept
aliases: [IFFT, 逆快速傅里叶变换]
---

# Inverse FFT

## 定义
将频域系数逆变换回时/空域的算法。

## 核心要点
1. 1D FFT 的逆运算，通常取实部 $\Re\{\cdot\}$
2. [[LaSt-ViT]] 用其把低通滤波后的频域信号还原为 patch 特征的低频重建

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[1D FFT]]
- [[Low-pass Filter]]
