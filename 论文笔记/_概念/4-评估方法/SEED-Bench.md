---
type: concept
aliases: [SEED-Bench, SEEDBench]
---

# SEED-Bench

## 定义
由腾讯 AI Lab 提出的全面评估多模态 LLM 的基准测试，包含 19K 道多选题，覆盖 12 个评估维度，同时评估图像和视频的空间/时间理解。

## 核心要点
1. 19K 道人工标注的多选题，覆盖空间理解（场景、对象、属性等）和时间理解（动作、顺序等）
2. 分为三个子集：All（全部）、Image（图像）、Video（视频）
3. Qwen-VL-Chat 在 All 上获 58.2（超越 LLaVA 的 32.7 近一倍），Image 上 65.4
4. 有趣发现：Qwen-VL 仅通过采样 4 帧即可迁移到视频任务（SEED-Bench Video 37.8）

## 代表工作
- [[Qwen-VL]]: 在 SEED-Bench 上大幅领先同期 LVLM
- [[MME]]: 另一 LVLM 评估基准

## 相关概念
- [[LVLM]]
- [[MME]]
- [[TouchStone]]
