---
type: concept
aliases: [MTP, 多 token 预测, multi-token prediction]
---

# Multi-Token Prediction (MTP)

## 定义
一种自回归预测策略，在主干网络生成当前 token 后，通过额外预测头 (MTP head) 并行或顺序预测多个未来 token。在 Qwen3-Omni 中，MTP 指 Talker 自回归预测第 0 层码本后，MTP 模块预测剩余残差码本。

## 核心要点
1. 层次化预测: 主干预测主要码本，MTP 模块预测残差码本
2. 固定步数自回归: MTP 为超轻量 Dense Transformer (80M)，低内存带宽需求
3. 固定 KV cache 加速: 固定步数推理可复用 KV cache，实现低延迟
4. 支持批处理推理: 轻量设计适配高并发场景

## 代表工作
- [[Qwen3-Omni]]: Talker MTP 模块进行多码本残差预测

## 相关概念
- [[Residual Vector Quantization]]
- [[Multi-Codebook Speech Codec]]
