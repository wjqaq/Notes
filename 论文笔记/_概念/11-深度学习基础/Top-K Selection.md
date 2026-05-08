---
type: concept
aliases: [Top-K 选择, TopK]
---

# Top-K Selection

## 定义
从集合中选出数值最大的 $K$ 个元素及其索引的操作。

## 核心要点
1. [[LaSt-ViT]] 按通道独立做 TopK 得 [[Channel-wise Top-K Pooling|通道级选择]]
2. $K$ 是关键超参，一般取总数的 1/4
3. 硬 TopK 不可微，可用 [[Gumbel Top-K]] 变可微

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Channel-wise Top-K Pooling]]
- [[Gumbel Top-K]]
