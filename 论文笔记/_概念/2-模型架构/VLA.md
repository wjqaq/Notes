---
aliases: [Vision-Language-Action Model, 视觉语言动作模型]
tags: [concept, vla, robotics, embodied-ai]
created: 2026-05-18
---

# VLA（Vision-Language-Action Model）

## 定义

VLA 模型将多模态基础模型（VLM）适配为预测机器人动作的策略，从视觉观测和语言指令映射到低层动作。

**基本公式**：
$$
a_{i,t} = \pi_\theta(I_{i,t}, L_i)
$$

其中 $a_{i,t}$ 编码末端执行器位姿和夹爪状态。

## 核心挑战

- [[Embodiment Tax]]: 动作微调系统性地侵蚀 VLM 的多模态能力
- **表征瓶颈**: 单一编码器同时承担语义理解和视觉运动控制
- **数据稀缺**: 机器人动作数据远少于 VL 数据

## 主要方法

- **冻结 VLM** + 动作头（如 VLM4VLA）
- **VL 联合训练**（如 ChatVLA、$\pi_{0.5}$+KI）
- **架构分离**（如 [[UAM]] 的 Dorsal Expert）

## 代表工作

- [[RT-2]]: 首创 VLA 范式
- [[OpenVLA]]: 开源 VLA
- [[π0]]: Flow Matching + MoT 的 VLA
- [[UAM]]: 双流架构 VLA，保留 >95% VLM 能力
