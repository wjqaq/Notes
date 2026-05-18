---
type: concept
aliases: [Constitutional AI, 宪法式AI, CAI]
---

# Constitutional AI

## 定义
一种基于预定义原则（宪法）指导 LLM 生成对齐响应的训练方法，通过让模型自我批评和自我改进来减少对人工反馈的依赖。

## 核心要点
1. 定义一套"宪法"原则（应遵循的和应避免的行为准则）
2. 让 LLM 根据宪法生成对齐的和偏离的响应作为偏好数据
3. 大幅减少人工标注需求，实现可缩放的对齐
4. 由 Bai et al. (2022) 在 Anthropic 提出

## 代表工作
- [[Qwen2]]: 后训练数据构建中用于安全与价值观对齐
- [[Claude]]: Anthropic 使用 CAI 训练的核心方法

## 相关概念
- [[RLHF]]
- [[Direct Preference Optimization|DPO]]
- [[Supervised Fine-Tuning]]
