---
type: concept
aliases: [VLMerger, Vision-Language Projector, 视觉语言投影器, 视觉语言融合器]
---

# Vision-Language Merger

## 定义
连接视觉编码器（ViT）和大语言模型（LLM）的中间模块，负责将视觉 token 的特征维度映射到 LLM 的文本嵌入空间。

## 数学形式
$$
v' = \text{MLP}([v_{i,j}; v_{i+1,j}; v_{i,j+1}; v_{i+1,j+1}]), \quad \forall (i,j) \in \{0,2,4,...\}
$$

## 核心要点
1. 最常见的实现是单层或多层 MLP，将 ViT 输出维度投影到 LLM 嵌入维度
2. Qwen2-VL 系列额外引入 $2 \times 2$ token 压缩：先拼接相邻 4 个 patch，再经 MLP 投影到 LLM 维度
3. 压缩使视觉 token 数减少 4 倍，大幅降低 LLM 的计算开销
4. 不同工作有不同设计：LLaVA 用单层线性，Qwen 系列用两层 MLP+压缩，InternVL 用 Pixel Shuffle

## 代表工作
- [[Qwen2-VL]]: 两层 MLP + $2 \times 2$ 压缩
- [[Qwen2.5-VL]]: 继承相同设计
- [[LLaVA]]: 单层线性投影

## 相关概念
- [[Vision Transformer]]
- [[MLP]]
- [[LVLM]]
