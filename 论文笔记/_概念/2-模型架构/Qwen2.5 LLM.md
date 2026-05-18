---
type: concept
aliases: [Qwen2.5-72B, Qwen2.5]
---

# Qwen2.5 LLM

## 定义
阿里通义千问团队发布的大语言模型系列，是 Qwen2 的升级版，为 Qwen2.5-VL 提供 LLM 骨干网络初始化权重。

## 核心要点
1. 相较于 Qwen2，在知识、推理、数学和代码能力上全面提升
2. 使用 1D RoPE 位置编码，词汇量 151646
3. Qwen2.5-VL 的 LLM 组件直接使用其预训练权重初始化
4. 提供多种规模（0.5B 到 72B），Qwen2.5-VL-72B 使用 80 层 8192 维

## 代表工作
- [[Qwen2.5-VL]]: LLM 组件基于 Qwen2.5-72B

## 相关概念
- [[RoPE]]
- [[LLM]]
- [[SwiGLU]]
- [[RMSNorm]]
