---
type: concept
aliases: [PaliGemma]
---

# PaliGemma

## 定义
Google 的 3B 参数多模态 VLM，在 TIPS/TIPSv2 中用于生成中等粒度的合成图像描述。

## 核心要点
1. 基于 Gemma 2B + SigLIP 视觉编码器
2. 为 WebLI 图像生成合成描述，显著优于噪声 alt-text
3. 局限性：描述中等粒度，遗漏姿态、背景、物体关系等细节
4. TIPSv2 将其作为多粒度文本方案的基础层级

## 代表工作
- [[TIPSv2]]: 使用 PaliGemma 描述作为多粒度文本的基础层
- [[TIPS]]: 使用 PaliGemma 生成合成描述
- Beyer et al. (2024): PaliGemma 原始论文

## 相关概念
- [[Gemini Flash]]
- [[Multi-Granularity Caption Sampling]]
- [[SigLIP]]
