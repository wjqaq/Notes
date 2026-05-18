---
type: concept
aliases: [OpenCLIP, Open-CLIP]
---

# OpenCLIP

## 定义
[[CLIP]] 的开源复现版本，提供多种规模的 ViT 预训练权重，是许多视觉语言模型的视觉编码器初始化来源。

## 数学形式
双塔对比学习：$\mathcal{L} = -\log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_j \exp(\text{sim}(I_i, T_j)/\tau)}$

## 核心要点
1. Qwen-VL 使用 ViT-bigG 变体作为视觉编码器
2. 相比原始 CLIP，OpenCLIP 在更大规模的数据（LAION-2B）上训练
3. 提供 ViT-B/32, ViT-B/16, ViT-L/14, ViT-H/14, ViT-bigG/14 等多种规格

## 代表工作
- [[Qwen-VL]]: 用 ViT-bigG 初始化视觉编码器
- [[SigLIP]]: 改进的对比学习变体

## 相关概念
- [[CLIP]]
- [[Vision Transformer]]
- [[Contrastive Learning]]
