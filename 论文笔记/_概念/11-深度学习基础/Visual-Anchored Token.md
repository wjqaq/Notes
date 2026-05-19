---
type: concept
aliases: [Visual-Anchored Token, 视觉锚定token, vision-anchored tokens]
---

# Visual-Anchored Token

## 定义
LVLM 生成响应中那些对输入视觉信息高度敏感和依赖的 token，通常为描述视觉内容的实体名词、属性形容词等。这些 token 是幻觉的高发区域。

## 核心要点
1. 由于大规模文本预训练带来的语言先验，LVLM 可能忽略视觉信息而依赖语言统计规律，导致视觉锚定 token 发生幻觉
2. 视觉锚定 token 通常为名词和形容词（如物体名称、颜色、数量等），占响应的约 39%
3. 通过对比原图和加噪图下的 logits 差异可自动识别视觉锚定 token
4. 在训练中应给予视觉锚定 token 更高的关注权重

## 代表工作
- [[TPO]]: 通过自校准奖励识别并优化视觉锚定 token
- [[Visual Contrastive Decoding|VCD]]: 推理时通过对比解码减少视觉锚定 token 的幻觉

## 相关概念
- [[Visual-Anchored Rewards]]
- [[Hallucination]]
- [[Language Prior]]
- [[TPO]]
