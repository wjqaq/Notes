---
type: concept
aliases: [表示工程, RepE, Activation Engineering]
---

# Representation Engineering

## 定义
通过直接操纵 LLM 隐空间中的激活向量来控制模型行为的技术范式，包括添加/减去方向向量、编辑注意力头、调整残差流等。

## 数学形式

基本操作：对隐状态 $h$ 施加转向向量 $v$：

$$
h' = h + \alpha \cdot v
$$

## 核心要点
1. 基于观察：高层概念（真值、安全、情感）在隐空间中线性编码
2. 优势：无需微调即可控制行为
3. 关键缺陷（[[Detection-Correction Asymmetry]]）：无差别编辑破坏正确生成
4. [[PCNet]] 将表示工程退化为仅用于检测，纠正路由到 token 空间

## 代表工作
- [[ITI]]: 推理时激活干预，向隐状态加真值向量
- TruthX: 真值方向的激活编辑
- AdaSteer, SADI: 自适应激活转向
- [[PCNet]]: 指出表示工程的局限，提出流形保持替代方案

## 相关概念
- [[Residual Stream]]
- [[Detection-Correction Asymmetry]]
- [[Hallucination]]
- [[Contrastive Decoding]]
