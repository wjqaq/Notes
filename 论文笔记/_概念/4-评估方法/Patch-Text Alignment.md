---
type: concept
aliases: [patch-text dense alignment, 密集图文对齐]
---

# Patch-Text Alignment

## 定义
视觉-语言模型中 patch 级图像表示与文本嵌入之间的密集对齐能力，直接影响 zero-shot 语义分割、开放词汇检测等 dense prediction 任务。

## 核心要点
1. CLIP 等全局对比学习方法缺乏 patch-text 对齐，zero-shot 分割极弱
2. TIPSv2 通过 iBOT++ 大幅提升 patch-text 对齐（ADE150 +14.1 mIoU）
3. 有趣现象：旗舰大模型在 patch-text 对齐上往往不如小模型（Table 14）
4. 评估方式：zero-shot semantic segmentation (mIoU via cosine similarity)
5. 蒸馏可使学生模型的 patch-text 对齐超越教师模型

## 代表工作
- [[TIPSv2]]: iBOT++ 带来最大 patch-text 对齐增益
- [[TIPS]]: 首次结合对比学习与 patch-level 蒸馏
- [[SigLIP2]]: 大规模下对齐退化

## 相关概念
- [[iBOT++]]
- [[Zero-shot Semantic Segmentation]]
- [[Contrastive Learning]]
