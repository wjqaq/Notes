---
aliases: []
tags: [concept, benchmark, simulation, robot-manipulation]
created: 2026-05-18
---

# CALVIN

## 定义

语言条件长时程机器人操作基准 [Mees et al., 2022]，使用 Franka Emika Panda 机械臂在仿真环境中执行序列操作任务。

## 特点

- ABC → D split：训练在 A/B/C 环境，测试在 D 环境（泛化评估）
- 长时程：每 episode 需完成多个顺序子任务
- 语言条件：自然语言指令指定任务序列

## 在 VLA 评估中的使用

- [[UAM]] 使用 CALVIN ABC-D 评估 in-domain 动作准确率
- 多个 VLA 工作将其作为标准仿真测试

## 代表工作

- [[UAM]]: Action Accuracy 评估
- [[VLM4VLA]]: VLA 遗忘研究中的动作评估
