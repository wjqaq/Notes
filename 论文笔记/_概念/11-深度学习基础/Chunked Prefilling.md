---
type: concept
aliases: [分块 Prefill, chunked prefilling, 分块预填充]
---

# Chunked Prefilling

## 定义
一种流式推理优化技术，将长序列输入拆分为多个块逐步进行 prefill，而非等待全部输入后再一次处理。在多模态模型中，不同模块可异步 prefill 不同块以降低首 token 延迟。

## 核心要点
1. 音频和视觉编码器沿时间维度分块输出
2. Thinker 完成当前块 prefill 后，即时用其高层表征异步 prefill Talker，同时 Thinker 处理下一块
3. 显著降低 Thinker 和 Talker 的 Time-To-First-Token (TTFT)
4. 需要编码器支持分块输出和块状注意力机制

## 代表工作
- [[Qwen2.5-Omni]]: 首次实现分块 prefill 机制
- [[Qwen3-Omni]]: MoE + 分块 prefill 进一步降低首包延迟

## 相关概念
- [[First-Packet Latency]]
- [[Streaming Speech Synthesis]]
