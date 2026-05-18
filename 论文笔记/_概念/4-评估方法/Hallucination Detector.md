---
type: concept
aliases: [Hallucination Detector, 幻觉判别器]
---

# Hallucination Detector

## 定义
基于跨模态注意力模式的轻量级幻觉检测模型，通常是二层 MLP，用于判断给定的注意力模式是否表现出幻觉特征。

## 数学形式
$$D(\mathbf{A}) = D_{l_2}(D_{l_1}(\text{flatten}(\mathbf{A}))) \in \mathbb{R}^{2}$$

## 核心要点
1. 输入为 flatten 的跨模态注意力向量（维度 $d = L \cdot H \cdot N$）
2. 二层 MLP：hidden=128, output=2（非幻觉/幻觉二分类）
3. 在 MHSA 框架中，判别器提供 detector-guided loss 的监督信号
4. MHSA 训练中判别器以极小学习率微调（保持稳定监督）

## 代表工作
- [[DHCP]]: 首个基于跨模态注意力的幻觉检测器
- [[MHSA]]: 将鉴别器重用作 token-level 判别器和训练监督信号

## 相关概念
- [[Cross-Modal Attention]]
- [[多模态幻觉]]
- [[DHCP]]
