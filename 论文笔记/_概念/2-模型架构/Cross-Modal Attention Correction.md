---
type: concept
aliases: [跨模态注意力修正, Attention Steering, 注意力引导修正]
---

# Cross-Modal Attention Correction

## 定义
通过可学习的生成器对 LVLM 的跨模态注意力模式进行残差修正，将幻觉注意力模式引导至非幻觉模式，实现无需修改 LVLM 参数的幻觉抑制。

## 数学形式
$$
\mathbf{A}' = \mathbf{A} + \Delta\mathbf{A} = \mathbf{A} + G(\mathbf{A})
$$

其中 $\mathbf{A}$ 为原始跨模态注意力，$G$ 为轻量 MLP 生成器（三层，hidden=512），$\Delta\mathbf{A}$ 为学习到的修正量。

## 核心要点
1. 残差设计：只学习偏移量而非从头预测，保持预训练注意力结构
2. 样本自适应：通过数据驱动学习，不同于 OPERA/PAI 的固定启发式规则
3. 仅修正幻觉样本：87.7% 样本无需修正（$D$ 判断为非幻觉），ammortized overhead +0.43x
4. Token-level 扩展：支持生成式任务的逐 token 注意力修正

## 代表工作
- [[MHSA]]: 提出首个可学习的跨模态注意力修正框架
- [[DHCP]]: 跨模态注意力幻觉检测（前序工作，仅检测不能修正）

## 相关概念
- [[Cross-Modal Attention]]
- [[Attention Steering Generator]]
- [[Residual Learning]]
- [[Hallucination Detector]]
