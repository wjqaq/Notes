---
type: concept
aliases: [Object Hallucination, 对象幻觉]
---

# Object Hallucination

## 定义
LVLM 幻觉的一类：模型在输出中错误地声称图像中存在某个物体，或遗漏图像中实际存在的物体。

## 核心要点
1. 两大类：False Positive（声称不存在的物体）和 False Negative（遗漏存在的物体）
2. POPE 评测通过 Yes/No 问询检测对象幻觉
3. 可能由过度依赖语言先验（如"厨房通常有冰箱"）而非视觉证据引起
4. MHSA 通过修正注意力使模型更依赖实际视觉信息来减少对象幻觉

## 代表工作
- [[POPE]]: 对象幻觉的判别式评测基准
- [[CHAIR]]: 图像描述中的对象幻觉评测
- [[MHSA]]: 通过跨模态注意力修正同时缓解 FP 和 FN 对象幻觉

## 相关概念
- [[多模态幻觉]]
- [[Attribute Hallucination]]
- [[Relational Hallucination]]
