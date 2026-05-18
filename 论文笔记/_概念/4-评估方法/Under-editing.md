---
category: 评估方法
tags: [image-editing, evaluation, failure-mode]
related_works: ["Entity-Rubrics", "Over-editing"]
created: 2026-05-18
---

# Under-Editing（编辑不足）

编辑不足是图像编辑中的一种失败模式，指模型未能充分执行指令要求的修改，导致输出图像与原始上下文图像过于相似。

## 特征

- 模型未能捕获抽象指令中的潜在需求
- 在开源模型中尤为常见（Entity-Rubrics 论文发现为开源模型的主要失败模式）
- 常导致 Preservation 指标被不真实地抬高（因为图像几乎没变）

## 与 Over-editing 的权衡

抽象编辑中存在 Under-editing vs Over-editing 的根本性权衡：切换到抽象提示可以减少过度编辑（平均 -13.3%），但开源模型会因此滑向编辑不足。
