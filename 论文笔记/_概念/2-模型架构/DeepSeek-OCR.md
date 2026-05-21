---
type: concept
aliases: [DeepSeek OCR, Contexts Optical Compression]
---

# DeepSeek-OCR

## 定义
DeepSeek 提出的端到端视觉-文本压缩模型，将文档图像编码为少量视觉 token，利用 LLM 重建文本，实现高达 20x 的上下文压缩。

## 核心要点
1. **DeepEncoder** (~380M): SAM-base (80M) → 16x Conv Compressor → CLIP-large (300M)，多分辨率支持
2. **LLM Decoder**: DeepSeek3B-MoE-A570M，12 层（1 standard + 11 MoE），top-k=6 专家激活
3. 视觉编码器与 LLM 联合优化（非 frozen），导致与传统 VLM 的本质差异
4. 4 种分辨率模式：Tiny (64 tokens), Small (100), Base (256), Large (400)
5. Gundam 模式支持更高分辨率（1083 tokens）

## 代表工作
- [[RTPrune]]: 针对 DeepSeek-OCR 的 token 剪枝方法
- DeepSeek-OCR2: 后续版本，引入 Visual Causal Flow

## 相关概念
- [[Visual-Text Compression]]
- [[Token Pruning]]
- [[Mixture of Experts]]
