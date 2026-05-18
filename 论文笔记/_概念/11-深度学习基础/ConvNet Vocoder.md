---
type: concept
aliases: [Code2Wav, ConvNet Vocoder, 卷积声码器]
---

# ConvNet Vocoder (Code2Wav)

## 定义
一种基于纯卷积神经网络的轻量级波形重建模块，将离散语音编解码 token 转换为时域音频波形。替代了计算密集的 block-wise Diffusion Transformer (DiT)。

## 核心要点
1. 全卷积架构: 因果 ConvNet，仅关注左上下文，支持流式生成
2. 轻量级 (200M 参数): 比 DiT 大幅降低 FLOPs 和推理延迟
3. 批处理推理: 卷积架构享有广泛的硬件加速支持，可高效批处理
4. 流式解码: 首帧即时合成，无需等待块级上下文

## 代表工作
- [[Qwen3-Omni]]: Code2Wav 替代 Qwen2.5-Omni 的 block-wise DiT

## 相关概念
- [[Codec-based Speech Generation]]
- [[Multi-Codebook Speech Codec]]
- [[First-Packet Latency]]
