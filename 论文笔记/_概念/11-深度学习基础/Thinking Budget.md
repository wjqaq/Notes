---
type: concept
aliases: [Thinking Budget]
---

# Thinking Budget

## 定义
Qwen3 中的推理预算控制机制，用户可设定思考 token 的上限数量，模型在达到预算时自动截断思考过程并基于已有推理生成最终回复。

## 核心要点
1. 是 Thinking Mode Fusion 的自然涌现能力（非显式训练）
2. 思考 budget 越大性能越好，呈平滑缩放曲线
3. 实现方式：思考 token 数达到阈值时插入停止指令 "Considering the limited time by the user..."
4. 在数学、编码、STEM 任务上均观察到一致的缩放规律
5. 若输出长度扩展超过 32K，性能有望继续提升

## 代表工作
- [[Qwen3]]: 首次引入 Thinking Budget 机制

## 相关概念
- [[Thinking Mode Fusion]]
- [[Chain of Thought]]
- [[GRPO]]
