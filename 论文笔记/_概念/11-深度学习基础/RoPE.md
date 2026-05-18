---
type: concept
aliases: [旋转位置编码]
---

# RoPE

## 定义
Rotary Position Embedding 的简称，通过旋转矩阵将位置信息编码到 token 的注意力计算中，使注意力分数自然包含相对位置关系。

## 数学形式
对于位置 $m$ 的 query 向量 $q_m$ 和位置 $n$ 的 key 向量 $k_n$：
$$(R_m q_m)^T (R_n k_n) = q_m^T R_{n-m} k_n$$

其中 $R_m$ 是旋转角度与位置 $m$ 成正比的旋转矩阵。

## 核心要点
1. 注意力分数仅依赖于相对位置 $n-m$，具有平移不变性
2. 可通过调整旋转频率（RoPE theta）控制模型对不同位置范围的敏感度
3. 在 Qwen2.5-VL 中扩展为 MRoPE（多模态 RoPE），在 Qwen3-VL 中进一步使用 Dual Chunk Attention + RoPE 支持超长上下文

## 代表工作
- [[Qwen2.5-VL]]: 使用 MRoPE（RoPE 的多模态扩展）
- [[Qwen2.5 LLM]]: 使用标准的 1D RoPE
- [[Qwen3-VL]]: 使用 Interleaved MRoPE

## 相关概念
- [[MRoPE]]
- [[2D-RoPE]]
- [[Interleaved MRoPE]]
- [[YaRN]]
