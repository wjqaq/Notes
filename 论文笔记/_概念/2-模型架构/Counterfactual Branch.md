---
type: concept
aliases: [反事实分支, counterfactual reference]
---

# Counterfactual Branch

## 定义
SIRA 中通过后期层对图像token位置施加[[注意力掩码]]构造的内部分支，保留共享多模态上下文但缺乏对细粒度视觉证据的持续访问，形成[[语言先验]]主导的内部参考。

## 核心要点
1. 从边界层 $b$ 起，图像token不能作为 key 被其他 query 读取，也不能作为 query 产生文本流
2. 不改变 prompt、参数或输出头，仅移除后期视觉访问路由
3. 保持在模型自身的 token 流形上，避免离流形伪影
4. 残差漂移在实证上很小（输出 [[KL散度]] 仅 0.012）

## 代表工作
- [[SIRA]]: 提出反事实分支设计，用于 token 级对比
