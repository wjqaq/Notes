---
category: 1-训练方法
aliases: [RLHF-V]
---

# RLHF-V

基于细粒度人工校正反馈的 VLM 行为对齐方法。收集 1.4k 片段级 (segment-level) 幻觉校正数据，提出 Dense DPO 进行对齐。数据质量极高但依赖人工标注，成本不可扩展。

## 代表工作
- [[RLHF-V]]: RLHF-V (CVPR 2024)
- [[TPR]]: 沿用了其分解和评分思路，并自动化了数据策划流程
