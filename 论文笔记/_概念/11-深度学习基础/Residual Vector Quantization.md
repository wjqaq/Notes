---
type: concept
aliases: [RVQ, 残差向量量化, multi-codebook]
---

# Residual Vector Quantization (RVQ)

## 定义
一种级联向量量化方法，通过多级码本逐层量化前一层残差，实现高保真离散表示。广泛应用于神经音频编解码。

## 数学形式
给定输入向量 $x$，第 $k$ 层量化：
$$c_k = \arg\min_{c \in \mathcal{C}_k} \| r_{k-1} - e(c) \|, \quad r_k = r_{k-1} - e(c_k)$$
其中 $r_0 = x$，$r_k$ 为第 $k$ 层残差。

## 核心要点
1. 多码本表示使每帧音频可由多个离散 token 完整描述，提升声学细节保真度
2. Qwen3-Omni Talker 自回归预测第 0 层码本，MTP 模块生成残差码本
3. 相比单码本，多码本显著增强对多样化音色、副语言线索和声学现象的建模能力

## 代表工作
- [[Qwen3-Omni]]: Talker 多码本自回归语音生成

## 相关概念
- [[Multi-Token Prediction]]
- [[Codec-based Speech Generation]]
- [[Multi-Codebook Speech Codec]]
