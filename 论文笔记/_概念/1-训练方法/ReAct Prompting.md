---
type: concept
aliases: [ReAct, Reasoning and Acting, 推理与行动]
---

# ReAct Prompting

## 定义
一种将推理（Reasoning）和行动（Acting）交织在一起的 LLM 提示格式，模型可以生成思考过程、调用工具、接收工具返回结果并基于观察继续推理。

## 核心格式
```
Thought: <推理下一步该做什么>
Action: <调用哪个工具>
Action Input: <工具参数>
Observation: <工具返回结果>
... (重复 Thought/Action/Observation 循环)
Final Answer: <最终答案>
```

## 核心要点
1. 使 LLM 具有自主调用外部工具的能力
2. 交替推理和行动，比纯推理或纯行动更可靠
3. 可通过 Few-shot 示例或微调来使模型掌握此格式

## 代表工作
- [[Qwen]]: 通过 Self-Instruct + ReAct 格式训练，使模型在中文工具调用 benchmark 上超过 GPT-4
- [[Yao et al. 2022]]: 首次提出 ReAct 框架

## 相关概念
- [[Self-Instruct]]
- [[Code Interpreter]]
