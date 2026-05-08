---
type: concept
aliases: [低通滤波, 低通滤波器]
---

# Low-pass Filter

## 定义
在频域抑制高频分量、保留低频成分的滤波器。

## 核心要点
1. [[LaSt-ViT]] 使用高斯低通核 $\mathbf{g}$ 与 FFT 结果 [[Hadamard Product|逐元素相乘]]
2. 滤波后 [[Inverse FFT]] 得到信号的'平滑'近似
3. 残差（原值 - 低通值）= 高频细节

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[1D FFT]]
- [[Inverse FFT]]
- [[Frequency-aware Selective Aggregation]]
