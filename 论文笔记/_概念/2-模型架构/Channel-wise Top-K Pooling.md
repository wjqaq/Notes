---
type: concept
aliases: [通道级 Top-K 池化]
---

# Channel-wise Top-K Pooling

## 定义
对每个通道独立选取稳定性最高的 K 个 patch 取平均作为新 CLS 该通道分量的聚合方式。

## 核心要点
1. 避免传统池化被背景 patch 主导
2. 产出 [[Vote Count]] $v_i$——patch 在所有通道里被选中的次数，可作为前景分数
3. 最佳 K 约为总 patch 数的 1/4

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[LaSt-ViT]]
- [[Top-K Selection]]
- [[Vote Count]]
