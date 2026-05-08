---
type: concept
aliases: [LazyStrike ViT, 惰性聚合修复 ViT]
---

# LaSt-ViT

## 定义
Shi et al. 2026 提出的 CLS 聚合替代方案，通过 [[Frequency-aware Selective Aggregation|频域稳定性打分]] + [[Channel-wise Top-K Pooling|通道级 Top-K 池化]]把 CLS 锚定到前景 patch。

## 核心要点
1. 零新增参数，可在预训练 ViT 权重上即插即用
2. 跨有监督 / CLIP / DINO 三种范式 12 个基准一致涨点
3. 产出的 [[Vote Count]] 可直接用作无监督定位分数

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Vision Transformer]]
- [[Register Tokens]]
- [[Patch Score]]
- [[Frequency-aware Selective Aggregation]]
- [[Channel-wise Top-K Pooling]]
