---
type: concept
aliases: [二维旋转位置编码, 2D Rotary Position Embedding]
---

# 2D-RoPE

## 定义
将旋转位置编码（RoPE）从一维序列扩展到二维空间，通过为高度和宽度维度分别赋予独立的旋转频率来捕获空间位置关系。

## 数学形式
二维位置 $(h, w)$ 的 RoPE 编码：
$$\Theta_{2D} = [\theta_h \cdot h, \; \theta_w \cdot w]$$

其中 $\theta_h, \theta_w$ 分别为高度和宽度维度的旋转频率。

## 核心要点
1. 相比 1D RoPE，能有效建模图像中 patch 之间的二维空间关系
2. 在 Qwen2.5-VL 中与 Window Attention 配合使用
3. 可扩展为 3D（时间+高度+宽度）以处理视频，即 MRoPE

## 代表工作
- [[Qwen2.5-VL]]: ViT 中采用 2D-RoPE 作为视觉编码器位置编码
- [[Qwen2-VL]]: 在 MRoPE 中间接使用 2D 空间分量

## 相关概念
- [[RoPE]]
- [[MRoPE]]
- [[Vision Transformer]]
