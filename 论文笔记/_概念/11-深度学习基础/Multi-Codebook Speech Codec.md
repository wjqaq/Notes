---
type: concept
aliases: [多码本语音编解码, multi-codebook codec]
---

# Multi-Codebook Speech Codec

## 定义
使用多个并行或级联的码本量化语音信号，使每帧音频由多个离散 token 描述。相比单码本，多码本提供更高的表示容量，能更忠实地建模音色、韵律、副语言等声学细节。

## 核心要点
1. 每帧音频由 K 个码本 token 联合描述，信息容量为 $K \cdot \log_2 |\mathcal{C}|$ bits/frame
2. Qwen3-Omni 使用层次化预测: 主干预测第 0 层，MTP 预测残差
3. 12.5Hz 码率下，每 80ms 一帧，每帧 K 个 token
4. 左上下文解码: 因果 ConvNet 仅依赖已生成的 token，实现首帧即时输出

## 代表工作
- [[Qwen3-Omni]]: Talker 多码本 12.5Hz 流式生成
- [[Qwen2.5-Omni]]: 单码本方案 (前身)

## 相关概念
- [[Residual Vector Quantization]]
- [[Codec-based Speech Generation]]
- [[ConvNet Vocoder]]
