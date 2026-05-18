---
category: 评估方法
tags: [benchmark, image-editing, evaluation]
related_works: ["Entity-Rubrics", "VIEScore"]
created: 2026-05-18
---

# ComplexEdit

ComplexEdit 是一个通过 CoT 式指令生成实现复杂度可控的图像编辑基准。它使用离散的逐分数描述来引导 VLM 进行评估，但其 VLM 评估存在宽容偏差（leniency bias），难以严格区分高质量和中等质量编辑。

## 与 Entity-Rubrics 的关系

Entity-Rubrics 在评估实验中将 ComplexEdit 作为对比评估指标，发现其分数膨胀问题，从而更突显 Entity-Rubrics 细粒度实体级评估的必要性。
