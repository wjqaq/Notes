---
type: concept
aliases: [偏好数据集, 偏好数据]
---

# Preference Dataset (偏好数据集)

## 定义
用于偏好优化的数据集合，每条数据包含输入和一对响应（chosen 优于 rejected），为模型提供偏好学习信号。

## 数学形式
$$\mathcal{D} = \{(x, v, y_w, y_l)\}$$

其中 $x$ 为文本指令，$v$ 为图像输入（VLM 场景），$y_w$ 为 preferred (chosen) 响应，$y_l$ 为 rejected 响应。

## 核心要点
1. 偏好信号来源: 人工标注、AI 评判、扰动生成、自批评
2. [[Re-Align]] 引入双偏好数据集: 同时包含 $(x, v, y_w, y_l)$ 文本偏好和 $(x, v, v_l, y_w)$ 视觉偏好
3. Re-Align 的 rejected 响应生成: 策略性遮蔽 + 检索相似图诱导幻觉
4. 信号质量 > 数据量（Re-Align 仅用 11k 样本即取得最佳效果）

## 代表工作
- [[Re-Align]]: 检索增强的双偏好数据集构建
- POVID: 高斯噪声 + GPT-4V 生成幻觉
- CSR: Self-rewarding 迭代选择

## 相关概念
- [[Preference Optimization]]
- [[Direct Preference Optimization]]
- [[Hallucination]]
