---
type: concept
aliases: [原生动态分辨率, Dynamic Resolution]
---

# Native Dynamic Resolution

## 定义
一种视觉输入处理方法，模型根据图像的实际尺寸（而非固定归一化尺寸）动态生成对应长度的 token 序列，使模型能够原生感知空间尺度和物体真实大小。

## 核心要点
1. 边界框和点坐标直接使用输入图像的实际像素尺寸，无需归一化
2. 图像在输入 ViT 前缩放至 28 的倍数，然后以 stride=14 切分为 patch
3. 训练时按原始宽高比随机采样，增强对不同分辨率的泛化能力
4. 相较于相对坐标，绝对坐标能更好地保留物体的真实尺度和空间关系

## 代表工作
- [[Qwen2.5-VL]]: 首次将原生动态分辨率与绝对时间 MRoPE 结合
- [[Qwen2-VL]]: 引入动态分辨率但使用相对坐标
- [[Qwen3-VL]]: 继承并扩展此机制

## 相关概念
- [[Vision Transformer]]
- [[MRoPE]]
- [[Dynamic FPS Sampling]]
