---
type: concept
aliases: [Qwen-VL, 通义千问视觉]
---

# Qwen-VL

## 定义
阿里通义千问团队发布的第一代视觉语言模型，基于 Qwen1 LLM + ViT，是 Qwen2-VL 的直接前身。

## 数学形式
ViT (可学习绝对位置编码) → Cross-Attention Resampler → Qwen1 LLM

## 核心要点
1. 使用 Cross-Attention Resampler 压缩视觉 token（固定 256 个视觉 token）
2. ViT 使用可学习的绝对位置编码，仅支持固定 448x448 分辨率输入
3. Qwen2-VL 完全重写：移除 Cross-Attention Resampler、加入 M-RoPE、实现动态分辨率
4. 奠定了 Qwen 多模态系列的基础架构方向

## 代表工作
- [[Qwen2-VL]]: 第二代，架构大幅革新
- [[Qwen2.5-VL]]: 第三代，增强细粒度感知

## 相关概念
- [[Qwen2]]
- [[Vision Transformer]]
- [[MRoPE]]
