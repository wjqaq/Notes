---
type: concept
aliases: [Head-only Exponential Moving Average, 仅头部EMA]
---

# Head-only EMA

## 定义
一种内存高效的训练策略，仅对投影头做指数移动平均更新，视觉编码器权重学生和教师共享，由 TIPSv2（CVPR 2026）提出。

## 数学形式
$$h_t \leftarrow \lambda h_t + (1-\lambda) h_s$$

其中 $f_t = f_s$（视觉编码器不做 EMA）。

## 核心要点
1. 动机：CLIP 损失已防止编码器坍塌，不需要全模型 EMA
2. ViT-B 上减少 42% 训练参数
3. 性能与全模型 EMA 持平，部分任务甚至略优
4. 完全移除 EMA 会导致训练严重不稳定

## 代表工作
- [[TIPSv2]]: 首次提出并验证 head-only EMA

## 相关概念
- [[Exponential Moving Average]]
- [[Knowledge Distillation]]
- [[DINO]]
