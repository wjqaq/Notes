---
type: concept
aliases: [Codec-based Speech Generation, 编解码器语音生成, codec speech synthesis]
---

# Codec-based Speech Generation

## 定义
将语音生成分解为两步范式：(1) 从文本/多模态表征生成离散语音编解码 token，(2) 从 token 序列重建波形。使用神经音频编解码器 (codec) 作为中间表示。

## 核心要点
1. 离散 token 表示使语音生成可复用语言模型的训练和推理范式
2. 多码本 (multi-codebook/RVQ) 提供更丰富的声学表示，每帧语音由多个码本 token 描述
3. Qwen3-Omni 使用 12.5Hz 码率，Talker 自回归预测第 0 层码本 + MTP 生成残差码本
4. 轻量因果 ConvNet (Code2Wav) 替代计算密集的 block-wise DiT 做波形重建

## 代表工作
- [[Qwen3-Omni]]: Talker 多码本编解码语音生成
- MaskGCT: 掩码生成编解码 Transformer
- Seed-TTS: 零样本语音生成编解码系列

## 相关概念
- [[Residual Vector Quantization]]
- [[Multi-Codebook Speech Codec]]
- [[ConvNet Vocoder]]
