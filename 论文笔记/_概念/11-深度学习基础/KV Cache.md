---
type: concept
aliases: [KV缓存, key-value cache, KV cache]
---

# KV Cache

## 定义
Transformer 自回归解码时缓存已生成 token 的 Key 和 Value 张量，避免重复计算的技术。

## 核心要点
1. 自回归生成时，每个新 token 只需计算其自身的 Q、K、V，历史 token 的 K、V 从缓存读取
2. 将解码复杂度从 $O(n^2 d)$ 降至 $O(n d)$（$n$ 为序列长度）
3. 存储所有 Transformer 层中所有注意力头的 K 和 V 张量
4. [[LIME]] 等方法通过在 KV cache 上施加可学习扰动来控制模型行为
5. 显存占用随序列长度线性增长，长序列可能成为瓶颈

## 代表工作
- [[LIME]]: 推理时优化 KV cache 扰动以抑制多模态幻觉

## 相关概念
- [[Inference-time Optimization]]
- [[Layer-wise Relevance Propagation|LRP]]
