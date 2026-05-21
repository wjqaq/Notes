---
type: concept
aliases: [视觉文本压缩, vision-text compression, contexts optical compression]
---

# Visual-Text Compression

## 定义
利用视觉模态作为文本信息的压缩介质，将长文本编码为少量视觉 token 送入 LLM 处理，从而绕过长上下文的高计算成本。

## 数学形式
输入文档图像 $\mathbf{I}$ → 视觉编码器 $f_v$ → 少量视觉 token $\mathbf{T}_v$ → LLM 解码为文本：

$$\mathbf{T}_v = g(f_v(\mathbf{I})), \quad |\mathbf{T}_v| \ll |\text{raw text tokens}|$$

压缩比可达 20x。

## 核心要点
1. DeepSeek-OCR 是该范式的代表性工作，使用 SAM + CLIP 组合的 DeepEncoder
2. 视觉 token 虽已大幅压缩文本，仍存在冗余（背景、重复结构），可进一步进行 token 剪枝
3. 与 OCR 任务的结合：压缩后需保持文本重建的完整性和精度
4. LLM 推理加速的直接途径：token 越少 → attention 计算量越小（$O(n^2)$）

## 代表工作
- [[DeepSeek-OCR]]: Contexts Optical Compression，视觉-文本压缩的标杆
- [[RTPrune]]: 在 DeepSeek-OCR 基础上的二次 token 级压缩

## 相关概念
- [[Token Pruning]]
- [[DeepSeek-OCR]]
