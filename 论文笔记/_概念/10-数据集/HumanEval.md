---
type: concept
aliases: [HumanEval, 代码生成评测]
---

# HumanEval

## 定义
一个包含 164 道 Python 编程题的数据集，用于评估 LLM 的代码生成能力，采用 pass@k 作为评估指标。

## 核心要点
1. 每道题包含函数签名、docstring 和若干单元测试
2. 模型根据 docstring 补全函数体
3. pass@1 是主要指标：生成 1 个候选通过所有测试的概率
4. GPT-4 0-shot pass@1 达 86.6

## 代表工作
- [[Qwen]]: Qwen-14B 0-shot pass@1 达 32.3（基座）；Qwen-Chat-14B 达 43.9
- [[Code-Qwen]]: Code-Qwen-Chat-14B 达 66.4，逼近 WizardCoder-34B

## 相关概念
- [[MBPP]]
