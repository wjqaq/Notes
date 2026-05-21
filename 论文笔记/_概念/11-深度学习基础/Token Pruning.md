---
type: concept
aliases: [token剪枝, visual token pruning, token reduction]
---

# Token Pruning

## 定义
在 Vision-Language Models 推理时减少视觉 token 数量以加速计算，同时尽可能保持任务性能的技术。

## 数学形式
给定视觉 token 序列 $\mathbf{T}_v \in \mathbb{R}^{N \times D}$，通过重要性函数 $s(\mathbf{T}_v^k)$ 选择 $M < N$ 个 token：

$$\mathcal{T}_{\text{kept}} = \text{Top-M}_{k}\{s(\mathbf{T}_v^k)\}$$

## 核心要点
1. 按剪枝位置分类：视觉编码器端（如 VisionZip, DivPrune）、LLM 内部（如 FastV, FitPrune）、多阶段（如 SparseVLM）
2. 按选择准则分类：基于注意力、基于相似度/多样性、基于文本相关性、基于特征 norm
3. OCR 场景中 token 剪枝面临特殊挑战：信息密度高、视觉-语言对齐弱、固定 prompt 无区分力
4. 训练无关方法（training-free）可即插即用，无需重新训练

## 代表工作
- [[RTPrune]]: 两阶段 token 剪枝，ℓ2-norm 选择 + 最优传输合并，专为 DeepSeek-OCR 设计
- [[FastV]]: 基于 LLM 注意力分数剪枝，在 layer 2 后执行
- [[DivPrune]]: 最大化最小成对距离保留多样化 token 子集

## 相关概念
- [[Optimal Transport]]
- [[ℓ2-Norm Feature Selection]]
- [[Dynamic Pruning Ratio]]
- [[Visual-Text Compression]]
