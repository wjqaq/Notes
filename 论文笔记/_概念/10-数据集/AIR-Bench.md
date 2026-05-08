---
type: concept
aliases: [AIR-Bench, Audio Instruction Reasoning Benchmark]
---

# AIR-Bench

## 定义
大规模音频-语言理解评测基准，覆盖语音（Speech）、声音（Sound）、音乐（Music）三个领域，包含 19k+ 单选题。

## 核心要点
1. 使用 GPT-4 评估生成式回答的质量
2. 三个子域各有不同难度和侧重点
3. 被 [[LIME]] 用于验证音频幻觉缓解效果
4. 代表性模型：SALMONN、Qwen2-Audio

## 代表工作
- [[LIME]]: Speech 37.51→45.20 (SALMONN), 57.56→66.10 (Qwen2-Audio)

## 相关概念
- [[Audio-Language Model]]
- [[多模态幻觉]]
