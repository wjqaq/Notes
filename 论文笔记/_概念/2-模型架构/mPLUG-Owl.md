---
type: concept
aliases: [mPLUG Owl, mPLUG-Owl]
---

# mPLUG-Owl

## 定义
阿里达摩院提出的模块化多模态大语言模型，通过视觉编码器和抽象器模块将图像信息注入 LLM。

## 核心要点
1. 模块化设计：视觉编码器 + 视觉抽象器 + LLM，各部分可独立替换
2. 视觉抽象器使用可学习 query 将图像特征压缩为少量 token
3. Qwen-VL 的对比之一，在同级开源 LVLM 中性能领先
4. 后续演化为 mPLUG-Owl2、DocOwl 等

## 代表工作
- [[Qwen-VL]]: 同期对标
- [[LLaVA]]: 另一主流 LVLM

## 相关概念
- [[LVLM]]
- [[Cross-Attention]]
