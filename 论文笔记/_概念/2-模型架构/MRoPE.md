---
type: concept
aliases: [Multimodal Rotary Position Embedding, 多模态旋转位置编码]
---

# MRoPE

## 定义
[[Qwen2-VL]] 提出的统一多模态位置编码方案，将嵌入维度划分为时间(t)、水平(h)、垂直(w)三个子空间，分别分配不同的旋转频率，实现对文本和视觉（图像、视频）位置的统一建模。

## 数学形式
对于 $d$-维嵌入，划分为 3 个连续子空间：

$$
\text{dims} = [\underbrace{d/3}_{t}, \underbrace{d/3}_{h}, \underbrace{d/3}_{w}]
$$

每个子空间使用独立的 [[RoPE]] 频率基。

## 核心要点
1. 3D 位置编码：t(帧索引), h(水平像素), w(垂直像素)
2. 通过不同频率基区分时空维度
3. 局限性：频率谱不平衡导致长视频理解退化（见 [[Interleaved MRoPE]]）

## 代表工作
- [[Qwen2-VL]]: 首次提出 MRoPE
- [[Qwen3-VL]]: 改进为 Interleaved MRoPE

## 相关概念
- [[Interleaved MRoPE]]
- [[RoPE]]
