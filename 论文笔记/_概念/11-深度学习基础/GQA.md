---
type: concept
aliases: [Grouped Query Attention]
---

# GQA

## 定义
Grouped Query Attention，将多个 Query head 分组共享同一组 Key/Value head，在 Multi-Head Attention 和 Multi-Query Attention 之间取得性能与效率的平衡。

## 数学形式
$$\text{GQA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$
其中每组 Query heads 共享相同的 K, V，Q:KV head 比例可以是 4:1、8:1 等。

## 核心要点
1. 减少 KV cache 大小，降低推理内存占用
2. 相比 MQA 保留更多注意力表达能力
3. Qwen3 各尺寸模型采用不同的 Q/KV head 比例（从 16/8 到 64/4）

## 代表工作
- [[Qwen3]]: Dense 和 MoE 模型均使用 GQA
- [[Qwen2.5]]: 引入 GQA 的前代模型

## 相关概念
- [[Multi-Head Attention]]
- [[QK-Norm]]
- [[KV Cache]]
