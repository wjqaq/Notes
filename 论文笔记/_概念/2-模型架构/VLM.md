---
type: concept
aliases: [Vision Language Model, 视觉语言模型, 多模态大模型, VLLM]
---

# VLM (Vision Language Model)

## 定义
将视觉编码器与大语言模型结合的模型架构，能同时理解图像和文本输入并生成文本输出。

## 数学形式
给定多模态输入 $(x, v)$（文本指令 $x$ + 图像 $v$），VLM 自回归生成响应：
$$y = [y_1, \dots, y_m] \sim \pi(\cdot|x, v)$$

## 核心要点
1. 典型架构: Vision Encoder $\to$ Projector $\to$ LLM Backbone
2. 视觉编码器常用 [[CLIP]]-ViT，将图像块编码为嵌入向量
3. Projector 将视觉嵌入对齐到 LLM 的文本嵌入空间
4. 训练分两阶段: 预训练（对齐视觉-文本空间）+ 指令微调（SFT）
5. 两大架构类型: Image-to-Text（编码器-解码器分离）和 Unified Model（统一编解码）

## 代表工作
- [[LLaVA]]: 首个开源高质量 VLM，基于 Vicuna + CLIP
- [[Re-Align]]: 检索增强 DPO 对齐缓解 VLM 幻觉
- Qwen-VL / Qwen2.5-VL: 通义千问多模态系列
- Janus-Pro: 统一多模态理解与生成
- BLIP-2: Q-Former 桥接视觉和语言

## 相关概念
- [[CLIP]]
- [[Direct Preference Optimization]]
- [[Hallucination]]
- [[Cross-Modal Alignment]]
