---
type: concept
aliases: [Massive Multitask Language Understanding, 多任务语言理解]
---

# MMLU

## 定义
一个大规模多任务语言理解评测基准，覆盖 57 个学科（STEM、人文、社科等），以多选题形式评估 LLM 的知识广度和深度。

## 核心要点
1. 57 个学科，约 14K 道四选一选择题
2. 常用 5-shot 或 0-shot 设置评估
3. GPT-4 达到 86.4%，作为当时最强 baseline

## 代表工作
- [[Qwen]]: Qwen-14B 5-shot 达 66.3，超越此前 13B SOTA；Qwen-14B-Chat 0-shot 达 64.6
- [[LLaMA 2]]: 70B 5-shot 达 69.8

## 相关概念
- [[C-Eval]]
