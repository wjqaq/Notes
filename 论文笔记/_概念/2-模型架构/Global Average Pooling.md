---
type: concept
aliases: [GAP, 全局平均池化]
---

# Global Average Pooling

## 定义
对所有 patch token 取逐通道平均得到全局表征的聚合方式。

## 核心要点
1. ViT 的两种主流聚合之一（另一个是 [[CLS Token]]）
2. 易受背景 patch 数量稀释——是 [[LaSt-ViT]] 想替代的对象
3. 计算便宜、无新增参数

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Vision Transformer]]
- [[CLS Token]]
