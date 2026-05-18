---
type: concept
aliases: [绝对时间编码, Temporal Grounding]
---

# Absolute Time Encoding

## 定义
MRoPE 的扩展技术，将时间维度的位置编码从帧序号改为实际时间戳（如秒级），使模型通过时间 ID 之间的间隔感知视频内容的节奏和精确时刻定位。

## 数学形式
对视频中的第 $t$ 帧，时间位置 ID 不是帧索引 $t$，而是基于实际经过的时间 $\Delta t$：
$$\theta_t \propto \Delta t_{\text{actual}}$$

## 核心要点
1. 无需额外计算开销，直接在 MRoPE 时间维度替换
2. 模型通过时间 ID 间隔的一致性学习时间对齐，适配不同 FPS
3. 支持秒级格式和 hmsf（时:分:秒:帧）格式的时间戳理解与输出
4. 在 Charades-STA 时间定位任务上 mIoU=50.9，远超前代

## 代表工作
- [[Qwen2.5-VL]]: 引入并实现绝对时间对齐的 MRoPE
- [[Qwen3-VL]]: 继承此设计

## 相关概念
- [[MRoPE]]
- [[Dynamic FPS Sampling]]
- [[RoPE]]
