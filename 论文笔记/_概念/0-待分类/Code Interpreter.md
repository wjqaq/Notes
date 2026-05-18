---
type: concept
aliases: [代码解释器, Python Code Interpreter]
---

# Code Interpreter

## 定义
一种赋予 LLM 编写和执行代码（通常是 Python）能力的技术，LLM 生成代码片段，执行引擎运行代码并返回结果，LLM 基于结果继续推理或生成最终答案。

## 核心要点
1. 使 LLM 具备精确的数学计算、数据分析和可视化能力
2. 对于需要多步规划的任务（如先检查 CSV 结构再画图），需要 LLM 具备规划能力
3. 评估维度包括代码可执行性（executability）和最终答案正确性（correctness）

## 代表工作
- [[Qwen]]: Qwen-14B-Chat 的代码可执行性接近 GPT-4（81.7% vs 86.8%）
- [[GPT-4]]: 内置 Code Interpreter 能力的标杆

## 相关概念
- [[ReAct Prompting]]
- [[LLM Agent]]
