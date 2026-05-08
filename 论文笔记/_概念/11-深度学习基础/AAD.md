---
type: concept
aliases: [Audio-Aware Decoding, 音频感知解码]
---

# AAD

## 定义
Audio-Aware Decoding，针对音频-语言模型设计的训练无关幻觉缓解解码策略。

## 核心要点
1. 据 [[LIME]] 论文描述，是当时唯一的训练无关音频幻觉缓解方法
2. 效果有限，在某些 benchmark 上甚至不如原始模型
3. [[LIME]] 中唯一的音频域 baseline

## 代表工作
- [[LIME]]: baseline 对比，LIME 在所有音频 benchmark 上超越 AAD

## 相关概念
- [[多模态幻觉]]
- [[Audio-Language Model]]
