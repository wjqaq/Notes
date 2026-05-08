---
type: concept
aliases: [Unified Multimodal Model, UMM, 统一多模态模型]
---

# Unified Multimodal Model

## 定义

能够同时处理多种模态（文本、图像、视频等）的理解和生成任务的单一模型架构，通常基于大型语言模型。

## 核心要点

1. 统一架构简化部署，减少模型数量
2. 通常结合 autoregressive LLM 和 diffusion/flow matching
3. 可分为 encoder-based 和 encoder-free 两类

## 代表工作

- [[Tuna-2]]: encoder-free UMM，直接处理像素
- [[BAGEL]]: encoder-based UMM
- [[Chameleon]]: Meta 的早期统一多模态模型
- [[Gemini]]: Google 的统一多模态模型

## 相关概念

- [[Vision Language Model]]
- [[Multimodal LLM]]
- [[Image Generation]]
