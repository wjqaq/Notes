---
category: 1-训练方法
aliases: [POVID]
---

# POVID (Preference Optimization for Vision-language models with Imperfect Data)

一种 rewriting-based 的 VLM 偏好对齐方法。强调 rejected 响应的重要性，通过扭曲图像和利用 GPT-4V 注入额外幻觉来生成高质量 rejected 响应，使用 17k 偏好数据微调 LLaVA-1.5-7B。

## 代表工作
- [[POVID]]: Zhou et al. (ICLR 2024 Workshop)
- [[TPR]]: 揭示了 POVID 等 rewriting 方法引入的幻觉分布与模型自身 failure mode 的差异
