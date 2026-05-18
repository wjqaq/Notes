---
type: concept
aliases: [原生动态分辨率, NDR]
---

# Naive Dynamic Resolution

## 定义
Qwen2-VL 提出的图像分辨率处理策略：移除 ViT 的绝对位置编码并改用 [[2D-RoPE]]，使视觉编码器能原生接受任意分辨率的输入，无需切分、填充等预处理。

## 数学形式
$$
\text{tokens} = \text{ViT}_{2\text{D-RoPE}}(\text{resize}(I, H', W')), \quad H' \bmod P = 0, \; W' \bmod P = 0
$$

## 核心要点
1. "Naive" 相对于 LLaVA-NeXT AnyRes 的网格切分方案——更简单、更直接
2. ViT 使用 2D-RoPE 后天然支持变长序列，无需固定输入尺寸
3. 输入图像仅需缩放到 patch_size（14）的整数倍，零 padding 浪费
4. 推理时根据图像实际分辨率自适应产生 token 数量
5. 高分辨率图像产生更多 token 保留细节，低分辨率使用更少 token 提高效率

## 代表工作
- [[Qwen2-VL]]: 首次提出 Naive Dynamic Resolution
- [[Qwen2.5-VL]]: 进一步结合 [[Window Attention]] 降低高分辨率的计算成本

## 相关概念
- [[2D-RoPE]]
- [[Vision Transformer]]
- [[Native Dynamic Resolution]]
