---
type: concept
aliases: [动态FPS采样, Dynamic Frame Rate Sampling]
---

# Dynamic FPS Sampling

## 定义
在视频理解训练中动态采样不同的帧率（FPS），使模型能够适应不同节奏的视频内容，而非固定帧率处理。

## 核心要点
1. 训练时动态调整 FPS，实现训练数据中 FPS 的均衡分布
2. 与绝对时间编码结合，通过时间 ID 间隔感知视频节奏
3. 支持从数秒到数小时的视频输入
4. 每视频最多分析 768 帧，总视频 token 不超过 24,576

## 代表工作
- [[Qwen2.5-VL]]: 首次在 LVLM 中结合动态 FPS 与绝对时间编码

## 相关概念
- [[Native Dynamic Resolution]]
- [[Absolute Time Encoding]]
- [[Long Video Understanding]]
