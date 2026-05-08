---
type: concept
aliases: [投票数, patch vote]
---

# Vote Count

## 定义
[[Channel-wise Top-K Pooling]] 中 patch $i$ 在所有通道的 Top-K 集合里被选中的总次数 $v_i$。

## 核心要点
1. $v_i \in \{0, \ldots, D\}$，越大越可能是前景
2. 可阈值化直接得到无监督前景 mask
3. 是 [[Unsupervised Object Discovery]] 的核心信号

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Channel-wise Top-K Pooling]]
- [[LaSt-ViT]]
- [[Unsupervised Object Discovery]]
