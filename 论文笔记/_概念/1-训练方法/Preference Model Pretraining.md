---
type: concept
aliases: [PMP, 偏好模型预训练]
---

# Preference Model Pretraining

## 定义
在训练 Reward Model 之前，先使用大规模比较数据对基础模型进行偏好预训练，使模型具备基本的偏好判断能力后再微调。

## 核心要点
1. 类似 LLM 预训练-微调范式：先在大量相对粗糙的比较数据上预训练，再用高质量标注数据微调
2. 比较数据格式：(query, response_a, response_b, preference)
3. PMP 后的模型在分布外偏好数据上也展示出良好泛化能力

## 代表工作
- [[Qwen]]: PMP 在多个分布外 human preference benchmark 上（Anthropic Helpful, OpenAI Summ., SHP, PRM800K）表现良好；RM 微调后在 Qwen 自有数据集上进一步提升
- [[Bai et al. 2022b]]: 提出 PMP 概念

## 相关概念
- [[Reward Model]]
- [[RLHF]]
