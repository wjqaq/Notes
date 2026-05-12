---
type: concept
aliases: [幻觉, 事实性错误, Hallucinations, Factuality Error, 多模态幻觉]
---

# Hallucination

## 定义
LLM/VLM 生成流畅自然但事实不正确、与输入不符或无依据的内容的现象。在 LLM 中表现为编造事实，在 VLM 中表现为生成与图像不一致的描述。

## 数学形式

在 [[PCNet]] 框架中，幻觉形式化为几何异常：

$$
\text{Hallucination} \iff \mathcal{S}_{\text{NLL}}(z) = -\log \mathcal{C}_{\text{root}}(z) \geq \tau
$$

即隐状态投影偏离 [[Contrastive Manifold|事实流形]] 进入低密度区域。

## 核心要点
1. 隐空间几何编码真实性：事实 token 在激活空间中聚类于特定区域
2. [[Detection-Correction Asymmetry]]：隐空间检测有效，但直接编辑破坏流利性
3. 检测方法：token 概率、语义熵、隐状态探针、密度估计
4. 纠正方法：表示工程、对比解码、检索增强生成

## 代表工作
- [[PCNet]]: 用精确密度估计检测和纠正 LLM 幻觉
- [[ITI]]: 推理时激活干预
- TruthX, DoLa, ICD: 各种幻觉缓解方法
- POPE, AMBER, MME: 多模态幻觉评测

## 相关概念
- [[Detection-Correction Asymmetry]]
- [[Anomaly Detection]]
- [[Contrastive Decoding]]
- [[Representation Engineering]]
- [[Factuality]]
