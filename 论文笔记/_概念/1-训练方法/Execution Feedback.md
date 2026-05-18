---
type: concept
aliases: [Execution Feedback, 执行反馈]
---

# Execution Feedback

## 定义
一种自动化数据合成方法：通过编译和执行 LLM 生成的代码/解决方案来评估其正确性，从而构建演示数据和偏好数据。

## 核心要点
1. 让 LLM 生成解决方案和相关测试用例
2. 通过编译和运行评估解决方案的正确性
3. 正确的解决方案作为演示数据，正确与错误的对比作为偏好数据
4. 也可用于评估指令遵循：LLM 生成验证函数检查响应是否满足约束

## 代表工作
- [[Qwen2]]: 后训练中用于代码任务的自动化数据合成

## 相关概念
- [[Rejection Sampling]]
- [[RLHF]]
- [[Constitutional AI]]
