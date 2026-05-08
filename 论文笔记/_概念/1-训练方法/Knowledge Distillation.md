---
type: concept
aliases: [KD, 知识蒸馏, teacher-student]
---

# Knowledge Distillation

## 定义
用大模型（教师）的输出来监督小模型（学生）的训练方法。TIPSv2 首次发现蒸馏可使学生 patch-text 对齐超越教师。

## 核心要点
1. 标准蒸馏：学生模仿教师的 logits/表示分布
2. TIPSv2 关键发现：蒸馏中的两个关键因素——
   - 移除 patch-level mask（对所有 token 监督）
   - 学生随机初始化（预训练初始化会消除收益）
3. 蒸馏出的 ViT-L 学生 zero-shot 分割超越 ViT-g 教师
4. 文本塔是否训练影响较小，核心在视觉编码器初始化方式

## 代表工作
- [[TIPSv2]]: 首次展示蒸馏对 patch-text 对齐的增益可超越教师
- Hinton et al. (2015): 知识蒸馏的原始提出
- [[DINO]]: 自蒸馏范式的代表作

## 相关概念
- [[Self-Distillation]]
- [[Exponential Moving Average]]
- [[DINO]]
