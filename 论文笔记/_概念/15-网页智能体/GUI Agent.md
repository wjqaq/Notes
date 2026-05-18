---
type: concept
aliases: [图形界面智能体, 图形用户界面智能体]
---

# GUI Agent

## 定义
基于多模态模型在图形用户界面上执行操作的智能体，包括感知（UI 元素定位）和决策（操作执行）两个核心环节。

## 核心要点
1. 感知层：截图理解 + UI 元素定位（按钮、输入框、图标的位置和功能）
2. 决策层：将移动端、网页端、桌面端操作统一为 function call 格式的共享动作空间
3. 推理过程通过人工+模型标注生成，防止过拟合 ground-truth 操作
4. Qwen2.5-VL 在 ScreenSpot Pro 上获 43.6%（远超次优 23.6%）
5. 无需 Set-of-Mark 辅助标记即可在真实环境中操作

## 代表工作
- [[Qwen2.5-VL]]: 引入 UI 定位 + 操作决策统一架构
- [[Qwen3-VL]]: 增强 Agent 能力
- [[Aguvis]]: 纯视觉 Agent

## 相关概念
- [[Visual Grounding]]
- [[Spatial Grounding]]
