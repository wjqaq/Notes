---
type: concept
aliases: [Decoding by Contrasting Layers, 层间对比解码]
---

# DoLa

## 定义
Decoding by Contrasting Layers，一种解码时幻觉缓解方法，通过对比 Transformer 成熟层和早期层的 logit 分布来增强事实性，无需外部知识或额外训练。

## 核心要点
1. 利用不同层对事实知识编码的差异
2. 属于 [[Contrastive Decoding]] 范式
3. 优势：无需训练，即插即用
4. 局限：对所有 token 无差别应用，腐化率 55.3%

## 代表工作
- Chuang et al. (2024): DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
- [[PCNet]]: 在其 token 空间安全性基础上加入精确密度门控

## 相关概念
- [[Contrastive Decoding]]
- [[Detection-Correction Asymmetry]]
- [[Hallucination]]
