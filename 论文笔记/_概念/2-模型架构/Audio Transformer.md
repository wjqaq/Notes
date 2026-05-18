---
type: concept
aliases: [AuT, Audio Transformer, AuT 编码器]
---

# Audio Transformer (AuT)

## 定义
Qwen3-Omni 自研的注意力-编码器-解码器自回归音频模型，在 2000 万小时监督音频上训练，编码器以 12.5Hz 输出通用音频表征。

## 核心要点
1. 使用 Conv2D 将滤波器组特征 8 倍下采样至 12.5Hz token rate
2. 采用 flash attention + 动态窗口尺寸 (1-8 秒)，平衡实时 prefill 缓存与离线性能
3. 训练数据: 80% 中英伪标签 ASR + 10% 其他语言 ASR + 10% 音频理解
4. 编码器参数约 0.6B，替代 Whisper 成为 Qwen3-Omni 的音频前端
5. 支持分块输出 (chunked output)，适配流式推理

## 代表工作
- [[Qwen3-Omni]]: AuT 作为音频编码器首次使用

## 相关概念
- [[Chunked Prefilling]]
- [[Streaming Speech Synthesis]]
