---
type: concept
aliases: [Strong-to-Weak Distillation, Knowledge Distillation]
---

# Strong-to-Weak Distillation

## 定义
一种知识蒸馏范式，使用更大更强的教师模型指导较小较弱的学生模型训练，在 Qwen3 中用于轻量模型的 post-training，分为 off-policy 和 on-policy 两阶段。

## 核心要点
1. Off-policy: 教师模型生成 thinking 和 non-thinking 输出，学生学习推理和模式切换
2. On-policy: 学生自生成序列，与教师 logits 对齐（最小化 KL 散度）
3. 蒸馏效果优于 RL：仅需 1/10 GPU 小时且 pass@64 更高
4. 学生保留探索能力（pass@64 提升），不会像 RL 那样收敛到单一模式
5. Qwen3 的 0.6B-30B-A3B 全部通过蒸馏获得

## 代表工作
- [[Qwen3]]: 轻量模型的核心训练方法
- [[DeepSeek-R1]]: 使用蒸馏将推理能力迁移到小模型

## 相关概念
- [[Knowledge Distillation]]
- [[GRPO]]
- [[Reinforcement Learning]]
- [[Thinking Mode Fusion]]
