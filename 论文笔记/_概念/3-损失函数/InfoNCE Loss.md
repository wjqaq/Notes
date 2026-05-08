---
type: concept
aliases: [InfoNCE, 信息噪声对比估计, contrastive loss, CLIP loss]
---

# InfoNCE Loss

## 定义
Information Noise-Contrastive Estimation，对比学习的核心损失函数，最大化正对互信息的下界，同时推开负对。

## 数学形式
$$\mathcal{L}_{InfoNCE} = -\frac{1}{2B}\sum_{k=1}^{B} \left[ \log\frac{\exp(s(I_k, T_k)/\tau)}{\sum_{j}\exp(s(I_k, T_j)/\tau)} + \log\frac{\exp(s(T_k, I_k)/\tau)}{\sum_{j}\exp(s(T_k, I_j)/\tau)} \right]$$

## 核心要点
1. 双向对称形式：I→T 和 T→I 两个方向
2. 温度参数 $\tau$ 控制分布的锐度
3. 依赖大 batch size 提供足够的负样本
4. CLIP/TIPS/TIPSv2 的核心损失组件

## 代表工作
- [[CLIP]]: 首次大规模使用 InfoNCE 做图文对比学习
- [[TIPSv2]]: 双 CLS 变体，分别对应不同粒度的文本
- CPC (Oord et al., 2018): InfoNCE 的原始提出

## 相关概念
- [[CLIP]]
- [[Contrastive Learning]]
- [[SigLIP]]
