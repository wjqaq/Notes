---
type: concept
aliases: [模态相关性, modality relevance score, 模态贡献度]
---

# Modality Relevance

## 定义
通过 [[Layer-wise Relevance Propagation|LRP]] 计算的感知模态 token（图像 patch / 音频帧）对模型输出的累积贡献度。

## 数学形式

模态 relevance:

$$
\Phi_M = \sum_{i \in M} \Phi_i
$$

对比优化目标:

$$
\mathcal{L}_{\text{rel}} = -\frac{1}{|M|} \sum_{i \in M} \log \frac{\exp(\Phi_{i,\Delta} / \tau)}{\sum_{k \in M \cup T} \exp(\Phi_{k,\Delta} / \tau)}
$$

## 核心要点
1. 将每个输入 token 的 relevance $\Phi_i$ 按模态聚合，得到模态级别的利用度量
2. [[LIME]] 的核心发现：文本 token 的 relevance 远高于感知 token，导致幻觉
3. 通过对比损失最大化模态 token 在全体 token 中的 relevance 占比
4. 温度参数 $\tau$ 控制 softmax 锐度

## 代表工作
- [[LIME]]: 第一个明确将 modality relevance 作为优化目标的工作

## 相关概念
- [[Layer-wise Relevance Propagation|LRP]]
- [[Modality Reliance]]
- [[Spatial Grounding]]
