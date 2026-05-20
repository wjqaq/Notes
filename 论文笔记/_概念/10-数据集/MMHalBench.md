---
type: concept
aliases: [MMHal-Bench, MMHal]
---

# MMHalBench

## 定义
用于评估[[大视觉语言模型]]幻觉的多维度基准，包含多种问题类型（物体属性、关系、上下文推理等），使用 GPT-4V 作为评判。

## 核心要点
1. 超越传统 caption 评测，覆盖多样化问答格式
2. 使用 GPT-4V 自动评分，包含 accuracy 和 detailedness 等维度
3. [[FLB]] 实验中验证了该方法在多样化任务上的泛化能力

## 代表工作
- (Sun et al., 2024): MMHalBench 提出
- [[FLB]]: 在此基准上取得 2.230 分，超越 baseline (1.944) 和 VCD (2.098)

## 相关概念
- [[CHAIR]]
- [[AMBER]]
- [[物体幻觉]]
- [[ConvBench]]
