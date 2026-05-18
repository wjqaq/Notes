---
type: concept
aliases: [Qwen-7B, 通义千问]
---

# Qwen

## 定义
阿里云通义千问团队开发的大语言模型系列，Qwen-VL 的 LLM 基座。

## 数学形式
Decoder-only Transformer，架构类似 LLaMA，使用 RoPE、RMSNorm、SwiGLU 激活函数。

## 核心要点
1. Qwen-7B 是第一代，Qwen-VL 用其中间 checkpoint 初始化 LLM
2. 后续演进：Qwen2 -> Qwen2.5 -> Qwen3，持续扩大参数和提升能力
3. 是多模态模型 Qwen-VL、Qwen-Audio 等的语言基座

## 代表工作
- [[Qwen-VL]]: 基于 Qwen-7B 的视觉语言模型
- [[Qwen2]]: 第二代 LLM，更强的基座能力
- [[Qwen3]]: 第三代，引入 MoE 架构

## 相关概念
- [[LLM]]
- [[Qwen2.5 LLM]]
- [[RoPE]]
- [[RMSNorm]]
- [[SwiGLU]]
