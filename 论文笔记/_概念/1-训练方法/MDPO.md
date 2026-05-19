---
type: concept
aliases: [MDPO, Multimodal DPO]
---

# MDPO

## 定义
Multimodal Direct Preference Optimization，一种对图像偏好也进行优化的 DPO 变体，将修改前后的图像分别作为正/负样本进行训练。

## 核心要点
1. 同时优化文本响应偏好和图像偏好
2. 将原始图像和修改后图像分别作为正/负视觉样本
3. 仅为响应级（sentence-level）奖励，缺乏 token 级精细度
4. 未特别关注视觉锚定 token

## 代表工作
- [[MDPO|Wang et al. 2024]]: mDPO: Conditional Preference Optimization for Multimodal Large Language Models

## 相关概念
- [[Direct Preference Optimization]]
- [[TPO]]
