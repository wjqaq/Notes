---
aliases: []
tags: [concept, benchmark, simulation, bimanual-manipulation]
created: 2026-05-18
---

# RoboTwin

## 定义

双臂机器人操作仿真基准 [Chen et al., 2025]，支持强域随机化，包含 50 个任务。

## 特点

- 双臂操作（bimanual）
- 域随机化：照明、纹理、物理参数
- 50 个任务覆盖多种技能（敲击、堆叠、倒物、按钮等）

## 在 VLA 评估中的使用

- [[UAM]] 选取 16 个最难任务（每个技能类别 1-2 个最低成功率任务）进行评估

## 代表工作

- [[UAM]]: 仿真 in-domain 动作评估
