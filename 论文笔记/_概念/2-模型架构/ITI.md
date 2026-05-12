---
type: concept
aliases: [Inference-Time Intervention, 推理时干预]
---

# ITI

## 定义
Inference-Time Intervention，一种推理时的幻觉纠正方法，通过线性探针识别注意力头中的"真值方向"，在推理时向残差流添加学到的转向向量来引导模型生成更真实的输出。

## 核心要点
1. 属于 [[Representation Engineering]] 范式
2. 对每个 token 无差别添加转向向量
3. 缺陷：腐化率高达 63.5%（[[Detection-Correction Asymmetry]]）
4. [[PCNet]] 将其作为对比基线，展示门控干预的优势

## 代表工作
- Li et al. (2023): Inference-Time Intervention: Eliciting Truthful Answers from a Language Model
- [[PCNet]]: 系统对比并指出 ITI 无差别编辑的局限

## 相关概念
- [[Representation Engineering]]
- [[Detection-Correction Asymmetry]]
- [[Hallucination]]
