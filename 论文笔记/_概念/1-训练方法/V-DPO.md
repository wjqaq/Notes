---
type: concept
aliases: [V-DPO, Vision-guided DPO]
---

# V-DPO

## 定义
Vision-guided Direct Preference Optimization，一种关注视觉锚定 token 的 DPO 变体，通过额外构建合成数据集来识别视觉相关 token 并施加 token 级优化。

## 核心要点
1. 关注视觉锚定 token 在幻觉中的作用
2. 需要额外构建合成数据集来支持 token 级优化
3. 仍依赖人工或半人工构建的数据
4. 与 TPO 同期工作，但 TPO 消除了对额外标注数据的需求

## 代表工作
- [[V-DPO|Xie et al. 2024]]: V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization

## 相关概念
- [[Direct Preference Optimization]]
- [[Visual-Anchored Token]]
- [[TPO]]
