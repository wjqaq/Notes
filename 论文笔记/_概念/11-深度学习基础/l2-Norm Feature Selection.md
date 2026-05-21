---
type: concept
aliases: [ℓ2-norm selection, embedding norm selection, feature norm pruning]
---

# ℓ2-Norm Feature Selection

## 定义
用特征向量的 ℓ2-norm 作为 token/特征重要性度量，norm 越大表示包含越多显著信息。

## 数学形式

$$C_k = \|\mathbf{T}_v^k\|_2 = \sqrt{\sum_{d=1}^{D} (T_{v}^{k,d})^2}$$

## 核心要点
1. 在 DeepSeek-OCR 中，视觉编码器与 LLM 联合优化，高 norm token 与 LLM 高注意力 token 高度相关（TIR 达 88.71%）
2. 相比 variance（可能遗漏整体偏移导致的高 norm）和 entropy（对密集分布不敏感），ℓ2-norm 更准确捕捉文本信息
3. 源于 ViT 中 register tokens 和 norm 与信息量的关系研究（Darcet et al., 2024）
4. 计算开销极低，无需额外模块

## 代表工作
- [[RTPrune]]: Stage 1 使用 ℓ2-norm 选择主导 token
- Darcet et al. (2024): Vision Transformers Need Registers — 发现高 norm token 对应 informative 区域

## 相关概念
- [[Token Pruning]]
- [[Register Tokens]]
