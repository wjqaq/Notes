---
aliases: [MoT]
tags: [concept, architecture, mixture-of-experts, routing]
created: 2026-05-18
---

# Mixture-of-Transformers（MoT）

## 定义

一种多专家并行路由架构，多个 Transformer 专家通过不同的注意力掩码实现信息交换，而非传统的 token 级 MoE 路由。

## 在 VLA 中的应用

- [[π0]]: 2-expert MoT（VLM 专家 + 动作专家）
- [[UAM]]: 3-expert MoT（语义专家 + Dorsal Expert + 动作专家）
- 优势：比 MLP 头更少遗忘（参数隔离减少模态干扰）

## 注意力掩码

各专家使用不同的注意力掩码控制 token 交互：
- ViT tokens → 语义专家
- VAE tokens → Dorsal Expert
- 动作专家 attend 所有 tokens

## 代表工作

- [[π0]]: 首创 VLA 中的 MoT 应用
- [[Bagel]]: 统一多模态 MoT 预训练
- [[UAM]]: 3-expert MoT 实现功能分叉
