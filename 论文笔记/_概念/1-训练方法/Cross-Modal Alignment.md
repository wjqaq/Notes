---
type: concept
aliases: [跨模态对齐, 图文对齐, Vision-Language Alignment]
---

# Cross-Modal Alignment (跨模态对齐)

## 定义
使视觉模态和文本模态在语义空间中保持一致和对齐的技术，确保模型生成的文本输出忠实于视觉输入。

## 核心要点
1. 对齐的两个层面:
   - 表示对齐: 视觉嵌入和文本嵌入在同一个语义空间中（如 [[CLIP]] 对比学习）
   - 生成对齐: 生成的文本内容与视觉输入保持一致（如偏好优化）
2. 对齐失败的表现: [[Hallucination|幻觉]]、图文不一致、虚假关联
3. [[Re-Align]] 通过 rDPO 同时优化文本偏好和视觉偏好，在生成层面强化跨模态对齐
4. 对齐税 (Alignment Tax): 过强对齐可能损害通用性能，是 RLHF 中已知的 trade-off

## 代表工作
- [[CLIP]]: 对比学习实现表示层跨模态对齐
- [[Re-Align]]: rDPO 在生成层面对齐视觉和文本
- [[Direct Preference Optimization|DPO]] 系列: 偏好信号引导对齐

## 相关概念
- [[Vision Language Model|VLM]]
- [[Hallucination]]
- [[CLIP]]
- [[Direct Preference Optimization]]
