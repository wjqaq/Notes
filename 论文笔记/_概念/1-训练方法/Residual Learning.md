---
type: concept
aliases: [Residual Learning, 残差学习, Residual Connection]
---

# Residual Learning

## 定义
学习目标函数的残差/偏移量而非完整映射的训练范式，通过 $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$ 形式简化优化。

## 核心要点
1. 起始点保留输入恒等映射，网络只学习修正项
2. 降低优化难度，缓解梯度消失
3. MHSA 中用于注意力修正：$\mathbf{A}' = \mathbf{A} + G(\mathbf{A})$

## 代表工作
- ResNet: 计算机视觉中的残差学习先驱
- [[MHSA]]: 注意力残差修正

## 相关概念
- [[Cross-Modal Attention]]
- [[Attention Steering Generator]]
