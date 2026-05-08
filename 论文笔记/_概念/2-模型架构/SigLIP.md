---
type: concept
aliases: [Sigmoid Loss for Language Image Pre-Training, SigLIP2]
---

# SigLIP

## 定义
用 sigmoid loss 替代 softmax 归一化的对比语言-图像预训练方法，将图文匹配建模为独立二分类问题。

## 数学形式
$$\mathcal{L}_{SigLIP} = -\frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{|B|}\log\frac{1}{1+\exp\big(z_{ij}\cdot(-t\cdot\mathbf{x}_i^\top\mathbf{y}_j + b)\big)}$$

其中 $z_{ij}=1$ 表示正对，$z_{ij}=-1$ 表示负对。

## 核心要点
1. 摆脱 softmax 对 batch size 的依赖，适合大 batch 训练
2. SigLIP2 加入 dense 对齐，但在大模型上 patch-text alignment 反而退化
3. 与 TIPSv2 对比：TIPSv2 的密集对齐显著优于 SigLIP2

## 代表工作
- [[TIPSv2]]: 在 zero-shot 分割上大幅超越 SigLIP2
- SigLIP (Zhai et al., 2023): 原始 sigmoid loss 论文
- SigLIP 2 (Tschannen et al., 2025): 加入定位和密集特征的升级版

## 相关概念
- [[CLIP]]
- [[InfoNCE Loss]]
- [[TIPSv2]]
