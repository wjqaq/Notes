---
type: concept
aliases: [HALC, Adaptive Focal-Contrast Decoding]
---

# HALC (Adaptive Focal-Contrast Decoding)

## 定义
通过自适应 focal-contrast 解码减少 LVLM 对象幻觉的推理时方法。

## 核心要点
1. 属于对比解码大类：对比不同输入的 logits
2. 引入自适应焦点机制调节对比强度
3. 推理时需多次前向传播

## 代表工作
- Chen et al. (2024, ICML): 提出 HALC

## 相关概念
- [[Contrastive Decoding]]
- [[VCD]]
- [[ICD]]
- [[多模态幻觉]]
