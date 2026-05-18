---
type: concept
aliases: [YaRN方法]
---

# YaRN

## 定义
Yet another RoPE extensioN — 一种 RoPE 位置编码的外推方法，通过缩放旋转频率和调整注意力温度，使模型在训练序列长度之外仍能保持位置编码的有效性。

## 核心要点
1. 用于将 RoPE-based 模型外推到更长上下文
2. 结合 NTK-aware 缩放和温度调节
3. Qwen3-VL 使用 YaRN 将 256K 上下文外推到 1M tokens（约 2 小时视频）

## 代表工作
- [[Qwen3-VL]]: 使用 YaRN 进行 Needle-in-a-Haystack 评估的外推

## 相关概念
- [[RoPE]]
- [[Interleaved MRoPE]]
