---
type: concept
aliases: [流式语音合成, streaming TTS]
---

# Streaming Speech Synthesis

## 定义
一种语音合成范式，模型逐帧或逐 token 生成语音并在生成过程中即时输出波形，无需等待完整序列生成完毕。对实时语音交互场景至关重要。

## 核心要点
1. 首帧即时合成: 生成第一个语音 token 后即开始波形输出，大幅降低首包延迟
2. 连续流式输出: 后续帧持续生成并输出，实现无间断语音流
3. RTF (Real Time Factor) < 1 是流式的必要条件
4. Qwen3-Omni 通过多码本自回归 + 轻量因果 ConvNet 实现流式语音合成

## 代表工作
- [[Qwen3-Omni]]: 234ms 首包延迟流式语音合成
- CosyVoice 2/3: 流式 TTS 系统

## 相关概念
- [[Real Time Factor]]
- [[First-Packet Latency]]
- [[Codec-based Speech Generation]]
