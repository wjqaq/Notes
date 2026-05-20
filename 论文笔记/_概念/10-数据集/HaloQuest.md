---
type: concept
aliases: [HaloQuest, HaloQuest Benchmark]
---

# HaloQuest

## 定义
多模态幻觉评测基准，通过三类子任务系统评估 MLLM 的幻觉程度。

## 核心要点
1. 三类子任务：
   - False Premise (错误前提): 问题包含不存在的物体（"图中是什么品种的狗？"— 实际没有狗）
   - Visually Challenging (视觉挑战): 需要精细视觉理解的问题
   - Insufficient Context (上下文不足): 图像信息不足以回答问题
2. 评测方式：Human Eval + Auto Eval 双重评估
3. 反概念场景 (Anti-concept) 特别考验模型的拒绝能力
4. MMGrounded-PostAlign 在此基准上 `<REJ>` token 带来最大增益

## 代表工作
- [[MMGrounded-PostAlign]]: 在 HaloQuest 上系统消融视觉定位各组件的贡献
- Woodpecker: 多模态幻觉后处理修正

## 相关概念
- [[多模态幻觉]]
- [[POPE]]
- [[负样本拒绝]]
