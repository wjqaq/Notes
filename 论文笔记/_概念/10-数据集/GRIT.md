---
type: concept
aliases: [GRIT, General Robust Image Task]
---

# GRIT

## 定义
通用鲁棒图像任务基准/数据集，提供大规模的 grounded captioning 标注（图像描述 + 名词短语对应的 bounding box），是训练 LVLM grounding 能力的核心数据源。

## 核心要点
1. 为每个 caption 中的名词短语提供对应的 bounding box 标注
2. Qwen-VL 使用 GRIT 的三个子任务：Grounding（生成带 grounding 的 caption）、Referring Grounding（根据描述找目标）、Grounded Captioning（生成描述并提供位置）
3. Qwen-VL 对 GRIT 数据进行贪心清洗，去除递归嵌套的 box 标注

## 代表工作
- [[Qwen-VL]]: grounding 数据主要来源
- [[Kosmos-2]]: 同样使用 GRIT 训练

## 相关概念
- [[Visual Grounding]]
- [[Grounded Captioning]]
- [[Referring Expression Comprehension]]
