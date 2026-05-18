---
type: concept
aliases: [三维卷积, 3D Conv]
---

# 3D Convolution

## 定义
在空间（H, W）和时间（T）三个维度上执行卷积核滑动的卷积操作，用于提取时空特征。

## 数学形式
$$
y_{t,i,j} = \sum_{k=0}^{K_t-1} \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} w_{k,u,v} \cdot x_{t+k, i+u, j+v} + b
$$

## 核心要点
1. 相比 2D 卷积增加时间维度，可捕获帧间运动信息
2. 深度可分离 3D 卷积（Separable 3D Conv）将空间和时间卷积分解，降低计算量
3. 在视频理解中，通常用 $1 \times 3 \times 3$ 卷积核（时间核 1，保持单帧空间特征）

## 代表工作
- [[Qwen2-VL]]: 在 ViT 中插入 3D 深度可分离卷积实现图像和视频的统一处理
- [[Qwen2.5-VL]]: 继承 3D 卷积处理视频

## 相关概念
- [[Vision Transformer]]
- [[MRoPE]]
