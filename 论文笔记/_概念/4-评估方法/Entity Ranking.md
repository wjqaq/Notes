---
type: concept
aliases: [实体排序]
---

# Entity Ranking

## 定义
Entity-Rubrics 评估框架的核心第二阶段，包含两个子阶段：Expected Transformation（基于指令为每个实体分配 Change/Optional Change/Preserve 预期状态）和 Execution Alignment（对比编辑后图像评估实际变换与预期的一致性）。

## 核心要点
1. Expected Transformation 代表 Identification 轴：在查看编辑结果之前确定"应该编辑什么"
2. Execution Alignment 将预期与结果对比：评估"实际做了什么"
3. 采用 precision-over-recall 策略：验证模型实际执行的编辑是否逻辑合理（而非穷举所有可能的有效解释）
4. 每个实体获得一个排序分数

## 代表工作
- [[Entity-Rubrics]]: 核心组件

## 相关概念
- [[Entity Detection]]
- [[Final Scoring]]
- [[Editing Degree of Freedom]]
