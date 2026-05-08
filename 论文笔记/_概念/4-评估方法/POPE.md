---
type: concept
aliases: [POPE benchmark, Polling-based Object Probing Evaluation, 基于轮询的对象探测评估]
---

# POPE

## 定义
评估多模态大模型对象级幻觉的 benchmark，通过二分类问题判断模型是否能在图像中正确判别对象存在性。

## 核心要点
1. 将幻觉评估转化为简单的 Yes/No 问题（"Is there a {object} in the image?"）
2. 三种负采样策略评估不同难度：
   - Random: 随机选不存在的对象
   - Popular: 选频繁出现在 MSCOCO 中但不在此图的对象
   - Adversarial: 选与图中对象高度相关但实际不在的对象
3. 评估指标：Accuracy、F1-Score、Yes Ratio
4. 数据集：MSCOCO（3000 样本）和 A-OKVQA（3000 样本）

## 代表工作
- [[LIME]]: Avg Acc 79.83 → 87.89, Avg F1 79.29 → 87.37
- (Li et al., 2023): POPE 原始提出

## 相关概念
- [[多模态幻觉]]
- [[CHAIR]]
