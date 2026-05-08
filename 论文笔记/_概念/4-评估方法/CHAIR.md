---
type: concept
aliases: [CHAIR metric, Caption Hallucination Assessment with Image Relevance]
---

# CHAIR

## 定义
评估图像描述生成中对象幻觉的指标，分对象级（CHAIR_I）和句子级（CHAIR_S）。

## 数学形式

$$
\text{CHAIR}_I = \frac{|\{\text{hallucinated objects}\}|}{|\{\text{all objects mentioned}\}|}
$$

$$
\text{CHAIR}_S = \frac{|\{\text{sentences with hallucinated object}\}|}{|\{\text{all sentences}\}|}
$$

## 核心要点
1. 基于 MSCOCO 的物体标注，检测生成描述中提及但图像中不存在的对象
2. CHAIR_I 衡量对象级幻觉比例，CHAIR_S 衡量句子级幻觉比例
3. 两个指标越低越好，反映模型对视觉输入的忠实度
4. 与 [[POPE]] 互补：POPE 测二分类判断，CHAIR 测开放式生成

## 代表工作
- [[LIME]]: CHAIR_S 从 52.0 降至 42.7（LLaVA-1.5-7B）
- (Rohrbach et al., 2018): CHAIR 原始提出

## 相关概念
- [[多模态幻觉]]
- [[POPE]]
