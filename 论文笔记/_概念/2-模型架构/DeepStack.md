---
type: concept
aliases: [多层视觉特征堆叠]
---

# DeepStack

## 定义
一种视觉-语言对齐增强机制，从 [[Vision Transformer|ViT]] 的多个中间层提取视觉特征，通过专用 merger 注入 LLM 对应层，实现从低级到高级的多层级视觉信息融合。

## 核心要点
1. 从 ViT 的浅层(低级纹理)、中层(部件结构)、深层(高级语义)三层提取特征
2. 每层配备专用 [[MLP]]-based merger 将特征投影为视觉 token
3. 以残差连接方式注入 LLM 的前三层隐藏状态
4. 不增加额外上下文长度，仅增加少量 merger 参数

## 代表工作
- [[Qwen3-VL]]: 首次将 DeepStack 应用于 VLM 跨层视觉特征注入
- Meng et al. (2024): 原始 DeepStack 概念，用于多尺度视觉输入的 token stacking

## 相关概念
- [[Vision Transformer]]
- [[Qwen3-VL]]
- [[Qwen2.5-VL]]
