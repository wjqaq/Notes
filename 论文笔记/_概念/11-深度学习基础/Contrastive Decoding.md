---
type: concept
aliases: [对比解码, CD, Latent Density Contrastive Decoding, LDCD]
---

# Contrastive Decoding

## 定义
一种解码策略，通过对比专家模型和业余模型的输出分布来惩罚不期望的生成，增强期望行为。在 LLM 中主要用于提升真实性和减少幻觉。

## 数学形式

标准对比解码：

$$
\text{Score}(x_t) = \log P_{\text{expert}}(x_t) - \lambda \cdot \log P_{\text{amateur}}(x_t)
$$

PC-LDCD 变体（密度惩罚版）：

$$
\text{Score}_{\text{LDCD}}(c_i) = \log P_{\text{LM}}(c_i \mid x_{<t}) - \beta_t \cdot \mathcal{S}_{\text{NLL}}(f_{\phi}(h_{t+1}^{(c_i)}))
$$

## 核心要点
1. 传统对比解码需要"业余"代理模型，可能引入偏差
2. PC-LDCD 用 [[Probabilistic Circuit]] 的精确 NLL 替代代理模型
3. 动态门控 $\beta_t$ 仅在隐状态异常时施加惩罚
4. 在离散 token 空间操作，避免隐空间编辑的语义崩溃

## 代表工作
- [[PCNet]]: PC-LDCD 用密度门控对比解码纠正幻觉
- [[DoLa]]: 成熟层与早期层 logit 对比
- [[ICD]]: 诱导幻觉对比解码
- [[SIRA]]: 共享前缀内部归因重构，模型内对比解码
- Li et al. (2023): Contrastive Decoding 基础框架

## 相关概念
- [[Probabilistic Circuit]]
- [[Negative Log-Likelihood]]
- [[Hallucination]]
- [[Detection-Correction Asymmetry]]
