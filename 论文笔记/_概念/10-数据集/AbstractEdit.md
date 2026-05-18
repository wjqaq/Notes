---
category: 数据集
tags: [benchmark, image-editing, abstract-editing]
related_works: ["Entity-Rubrics"]
created: 2026-05-18
---

# AbstractEdit

AbstractEdit 是首个专门评估抽象图像编辑的基准数据集。由 Mor Ventura 等人在 "Editor's Choice" 论文中提出。

## 规模与构成

- **测试集**: 470 个人工验证样本，覆盖 Physical、Logical、Emotional、Social 四个领域 12 个子类别
- **训练集**: 4,116 个自动生成样本
- **源图像**: 来自 [[Open Images v7]] 的 1,300 张复杂多实体场景
- 每个样本包含：上下文图像 + 抽象编辑指令 + 配对的显式编辑指令

## 生成方式

通过自动策展流水线生成：Persona 驱动的多样化指令 -> [[Gemini 2.5 Pro]] few-shot 生成配对指令 -> 作者人工验证测试集抽象性。

## 领域分布

- Logical (186): 包括 CommonsenseGoal、InsertionGoal、Temporal
- Social (141): 包括 Culture、Socio-Economic、Role、Genre/Narrative
- Physical (101): 包括 Season、Pose、POV/Composition、Size
- Emotional (41): Mood/Emotion
