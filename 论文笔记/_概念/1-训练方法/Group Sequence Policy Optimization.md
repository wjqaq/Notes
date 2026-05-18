---
type: concept
aliases: [GSPO, Group Sequence Policy Optimization]
---

# Group Sequence Policy Optimization (GSPO)

## 定义
Qwen 团队提出的强化学习方法，通过分组序列间的策略优化全面增强模型在各模态上的能力和稳定性。结合规则奖励和模型奖励。

## 核心要点
1. 规则奖励 (Rule-based Reward): 针对可验证任务（数学、代码、指令遵循），通过预定义规则高精度评估输出正确性
2. 模型奖励 (Model-based Reward): 针对无客观标准的任务，采用 LLM-as-a-Judge 协议（Qwen3 做通用评判，Qwen2.5-VL 做视觉评判），辅以参考答案
3. 覆盖全模态: 文本、图像、视频、音频
4. Qwen3-Omni 使用 GSPO 做后训练最后一阶段

## 代表工作
- [[Qwen3-Omni]]: GSPO 后训练增强多模态能力
- [[Qwen3]]: GSPO 后训练增强文本能力

## 相关概念
- [[Direct Preference Optimization]]
- [[Strong-to-Weak Distillation]]
