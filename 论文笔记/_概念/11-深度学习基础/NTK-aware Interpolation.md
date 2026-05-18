---
type: concept
aliases: [NTK-Aware RoPE, NTK-aware Scaled RoPE, NTK插值]
---

# NTK-aware Interpolation

## 定义
一种训练无关的 RoPE 上下文长度扩展方法，不等比缩放 RoPE 的不同频率维度以保留高频信息，避免简单的线性位置插值导致的性能损失。

## 数学形式
传统 PI (Position Interpolation) 将所有维度等比例缩放：

$$
\theta'_i = \theta_i \cdot \frac{L}{L'}
$$
NTK-aware 方法对不同频率维度采用不同缩放因子，高频维度保留更多原始信息：

$$
\theta'_i = \theta_i \cdot \left(\frac{L}{L'}\right)^{\alpha_i}
$$

## 核心要点
1. 解决简单 Position Interpolation 在高频信息上的损失
2. 动态 NTK-aware 按 chunk 动态调整缩放因子，避免严重性能退化
3. 无需额外训练，纯推理时应用

## 代表工作
- [[Qwen]]: 结合动态 NTK-aware 插值、LogN-Scaling 和 Window Attention 实现 8192+ 上下文
- [[LLaMA]] 社区: 广泛使用的长上下文扩展方法

## 相关概念
- [[RoPE]]
- [[LogN-Scaling]]
- [[Window Attention]]
- [[Length Extrapolation]]
