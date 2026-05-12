---
type: concept
aliases: [残差流, 残差连接流, Hidden Stream]
---

# Residual Stream

## 定义
Transformer 架构中各层输出的加性累积信号通道，每一层通过残差连接将其输出加到前序表示上，形成贯穿网络的"信息高速公路"。

## 数学形式

第 $l$ 层的隐状态更新：

$$
h^{(l)} = h^{(l-1)} + \text{Attention}(h^{(l-1)}) + \text{FFN}(h^{(l-1)} + \text{Attention}(h^{(l-1)}))
$$

## 核心要点
1. 隐空间中线性编码了语义、事实性等高级属性
2. 最后一层 residual stream 的隐状态 $h_{\text{last}}$ 常用于下游分析
3. [[PCNet]] 以 $h_{\text{last}}$ 为输入，投影后做密度估计
4. [[Representation Engineering]] 通过加减残差流向量控制行为

## 代表工作
- [[PCNet]]: 对 residual stream 的最后隐状态做密度估计
- Marks & Tegmark (2024): 真值方向在残差流中线性编码
- [[ITI]]: 在残差流中添加/减去真值向量

## 相关概念
- [[Representation Engineering]]
- [[Information Bottleneck]]
- [[Transformer]]
