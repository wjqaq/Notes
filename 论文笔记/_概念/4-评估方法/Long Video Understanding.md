---
type: concept
aliases: [长视频理解, Long-form Video Understanding]
---

# Long Video Understanding

## 定义
对时长从数十分钟到数小时的视频进行内容理解的能力，包括事件定位、时序推理和细粒度时间 grounding。

## 核心要点
1. Qwen2.5-VL 通过动态 FPS + 绝对时间 MRoPE 实现小时级视频理解
2. 每视频最多分析 768 帧，总视频 token 不超过 24,576
3. LVBench（长视频理解）上 Qwen2.5-VL 获 47.3，远超 GPT-4o 的 30.8
4. Charades-STA 时间定位 mIoU=50.9，超越 GPT-4o (35.7)
5. 支持秒格式和 hmsf（时:分:秒:帧）格式的时间戳输出

## 代表工作
- [[Qwen2.5-VL]]: 在 LVBench 和 MLVU 上显著超越 GPT-4o
- [[Qwen3-VL]]: 通过 Native Sparse Attention 进一步扩展视频长度

## 相关概念
- [[Dynamic FPS Sampling]]
- [[Absolute Time Encoding]]
- [[MRoPE]]
