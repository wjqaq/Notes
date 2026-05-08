---
type: concept
aliases: [Masked Image Modeling, MIM, 掩码图像建模]
---

# Masked Image Modeling

## 定义

一种自监督学习方法，通过随机遮蔽图像部分区域并让模型预测被遮蔽内容来学习视觉表示。

## 数学形式

$$
\mathcal{L}_{MIM} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|f(x)_{\text{mask}_i} - x_{\text{mask}_i}\|^2
$$

其中 $\mathcal{M}$ 为被遮蔽的位置集合。

## 核心要点

1. 强迫模型理解图像全局结构和语义
2. 可作为预训练或训练时的正则化
3. 在 encoder-free 模型中尤为重要，用于稳定高维像素空间训练

## 代表工作

- [[Tuna-2]]: 使用 masking 稳定像素空间训练
- [[MAE]]: 经典的 masked autoencoder 方法
- [[BEiT]]: 使用离散 token 预测的 MIM 方法

## 相关概念

- [[Self-Supervised Learning]]
- [[Vision Transformer]]
- [[Masked Language Modeling]]
