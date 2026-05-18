---
type: concept
aliases: [首包延迟, Time-to-First-Packet, TTPT, Time-to-First-Token, TTFT]
---

# First-Packet Latency

## 定义
从用户请求发出到系统返回第一个响应包（文本 token 或语音波形）的时间。对流式交互系统的用户体验至关重要。

## 核心要点
1. 包含预处理延迟 + Thinker TTFT + Talker TTFT + MTP 延迟 + Codec 延迟
2. Qwen3-Omni 冷启动首包延迟 234ms（单并发音频），通过 MoE + 分块 prefill + 多码本流式 + 轻量 ConvNet 实现
3. 高并发下延迟会增加，但 MoE 架构使其增长可控

## 代表工作
- [[Qwen3-Omni]]: 端到端 234ms 首包延迟

## 相关概念
- [[Real Time Factor]]
- [[Chunked Prefilling]]
- [[Streaming Speech Synthesis]]
