---
type: concept
aliases: [Sobel算子, Sobel edge detection, Sobel filter]
---

# Sobel Operator

## 定义
一种离散微分算子，用于计算图像强度函数的梯度近似值，在文档分析中特别擅长识别字符笔画和边界。

## 数学形式
水平与垂直 3x3 卷积核：

$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} +1 & +2 & +1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix} * I$$

梯度幅值：

$$G(i,j) = \sqrt{G_x(i,j)^2 + G_y(i,j)^2}$$

## 核心要点
1. 灰度转换使用 ITU-R BT.601 系数：$I = 0.299R + 0.587G + 0.114B$
2. 文档图像中文本区域呈现高梯度幅值（sharp transitions），背景/空白区域低幅值
3. 在 RTPrune 中用于估计每个 token patch 的文本密度 $\rho_k$
4. 阈值 $\tau$ 控制边缘检测敏感度（典型值 0.2）

## 代表工作
- [[RTPrune]]: 用 Sobel 算子估计文本密度 $\rho$，指导动态剪枝率计算

## 相关概念
- [[Dynamic Pruning Ratio]]
- [[Visual-Text Compression]]
