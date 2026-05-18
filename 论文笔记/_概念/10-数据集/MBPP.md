---
type: concept
aliases: [Mostly Basic Python Programming, Python基础编程]
---

# MBPP

## 定义
一个包含约 1000 道 Python 基础编程题的数据集，评估 LLM 的代码生成能力，题目偏基础但覆盖面广。

## 核心要点
1. 题目来自入门级编程任务，难度低于 HumanEval
2. 常用 3-shot 或 0-shot 评估
3. 评估方式与 HumanEval 类似，使用 pass@k

## 代表工作
- [[Qwen]]: Qwen-14B 3-shot pass@1 达 40.8；Code-Qwen-14B 达 51.4
- [[Code-Qwen]]: Code-Qwen-Chat-14B 达 52.4

## 相关概念
- [[HumanEval]]
