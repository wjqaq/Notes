---
category: 评估方法
tags: [image-editing, evaluation, failure-mode]
related_works: ["Entity-Rubrics", "Under-editing", "CLIP"]
created: 2026-05-18
---

# Over-Editing（过度编辑）

过度编辑是图像编辑中的一种失败模式，指模型为满足指令而对图像进行了过多的修改，破坏了原始上下文图像的重要内容。

## 特征

- 闭源模型的典型失败模式（Entity-Rubrics 论文发现）
- [[CLIP]] 等编码类指标倾向于奖励过度编辑，因为全局变化提升了图文相似度
- 长文本指令（显式指令）会加剧过度编辑，因为文本 token 主导了图像上下文

## 缓解

- 使用抽象（而非显式）提示可平均减少 13.3% 的过度编辑
- Entity-Rubrics 评估框架能有效区分意图驱动的变换和任意修改
