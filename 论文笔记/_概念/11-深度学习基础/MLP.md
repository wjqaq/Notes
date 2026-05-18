---
type: concept
aliases: [多层感知机, Multi-Layer Perceptron, 全连接网络, Fully Connected Network]
---

# MLP

## 定义
多层感知机（Multi-Layer Perceptron），由多个全连接层和激活函数堆叠而成的前馈神经网络，是最基础的神经网络架构之一。

## 数学形式
$$
\mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})
$$

其中 $\sigma$ 为激活函数（如 [[ReLU]]），$\mathbf{W}^{(l)}$ 和 $\mathbf{b}^{(l)}$ 为第 $l$ 层的权重和偏置。

## 核心要点
1. 万能近似器：单隐层 MLP 即可近似任意连续函数（给定足够宽度）
2. 轻量高效：作为小规模特征变换模块嵌入复杂架构（如注意力修正、分类头）
3. MHSA 中使用三层 MLP（hidden=512）作为跨模态注意力修正生成器

## 代表工作
- [[MHSA]]: 三层 MLP 作为注意力修正生成器
- [[DHCP]]: 二层 MLP（hidden=128）作为幻觉检测判别器

## 相关概念
- [[ReLU]]
- [[Residual Learning]]
