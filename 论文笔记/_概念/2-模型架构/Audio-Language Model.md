---
type: concept
aliases: [音频-语言模型, ALM, audio MLLM, 听觉大模型]
---

# Audio-Language Model

## 定义
以音频（语音/环境声/音乐）作为感知输入模态，结合语言模型进行理解和生成的多模态大模型。

## 核心要点
1. 典型架构：音频编码器（Whisper/CLAP）+ 适配器 + LLM
2. 代表模型：SALMONN（Whisper large v2 + Vicuna）、Qwen2-Audio（Whisper large v3 + Qwen）
3. 核心挑战：时序定位精度、复杂声学场景理解、与语言模型的对齐
4. 幻觉问题：与视觉模型类似，模型可能忽略音频证据而依赖语言先验
5. 评测：[[AIR-Bench]]、Audio Hallucination QA

## 代表工作
- [[LIME]]: 首次将推理时幻觉缓解统一应用于音频和视觉模型

## 相关概念
- [[多模态幻觉]]
- [[AIR-Bench]]
