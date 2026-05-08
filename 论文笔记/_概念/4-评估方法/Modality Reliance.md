---
type: concept
aliases: [模态依赖, modality reliance score, 模态依赖度]
---

# Modality Reliance

## 定义
衡量模型在生成回答时依赖感知模态（vs 文本）的程度，通过感知 token relevance 在总 token relevance 中的占比量化。

## 核心要点
1. $M_{\text{reliance}} = \Phi_M / (\Phi_M + \Phi_T)$，即模态 relevance 占总 relevance 的比例
2. 高 modality reliance 表示模型更"看/听"感知输入，低值表示更依赖语言先验
3. [[LIME]] 的核心发现：原始模型的 modality reliance 往往很低（如 LLaVA-1.5 仅 0.10）
4. LIME 优化后所有模型的 modality reliance 均提升

## 代表工作
- [[LIME]]: 首次提出并优化 modality reliance 指标

## 相关概念
- [[Modality Relevance]]
- [[Spatial Grounding]]
- [[多模态幻觉]]
