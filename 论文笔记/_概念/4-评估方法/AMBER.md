---
type: concept
aliases: [AMBER benchmark, 多维幻觉评估]
---

# AMBER

## 定义
AMBER 是一个无需 LLM 的多维 LVLM 幻觉评估基准，同时测量幻觉率和描述覆盖度。

## 核心要点
1. 评估四个指标：CHAIR（物体幻觉）、Hal（一般幻觉）、Cog（认知幻觉）、Cover（描述覆盖度）
2. Hal 和 Cog 越低越好，Cover 越高越好
3. Cog 指标专门捕捉与[[语言先验]]最紧密耦合的认知幻觉
4. 联合报告 Hal/Cog 与 Cover 可以判断方法是否通过缩短/保守化回答来降低幻觉

## 代表工作
- [[SIRA]]: 在 AMBER 上将 Cog 降低 50%，同时保持/提升 Cover
