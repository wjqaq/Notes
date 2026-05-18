---
aliases: [世界模型, visual dynamics prediction, 视觉动力学预测]
tags: [concept, world-model, visual-dynamics, robotics]
created: 2026-05-18
---

# World Model（世界模型）

## 定义

学习预测环境状态演变的模型，在机器人领域通常表现为预测未来视觉观测（视觉动力学预测）。

## 在 VLA 中的应用

- 作为策略学习的辅助信号：预测目标观测 $\hat{I}_{t+1}$，匹配真实 $I_{t+1}$
- 帮助模型理解动作如何影响场景
- 在 [[UAM]] 中作为 Dorsal Expert 的监督信号 $\mathcal{L}_{wm}$

## 关键洞察

仅凭容量或语义先验不足以使第二条通路成为有效的控制路径——需要匹配的动力学监督信号驱动其进行独立的视觉推理。

## 代表工作

- [[UAM]]: 视觉动力学损失驱动 Dorsal Expert 的背侧功能
- [[BagelVLA]]: 交织语言规划、视觉预测和动作生成
- [[UniPi]]: 文本引导视频生成学习通用策略
- [[Motus]]: 统一潜在动作世界模型
