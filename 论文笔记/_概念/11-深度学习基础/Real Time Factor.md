---
type: concept
aliases: [RTF, 实时因子, real time factor]
---

# Real Time Factor (RTF)

## 定义
衡量流式语音系统处理效率的核心指标，定义为生成单位时长音频所需的计算时间与音频时长之比。RTF < 1 表示可实时流式输出。

## 数学形式
$$\text{RTF} = \frac{T_{\text{compute}}}{T_{\text{audio}}}$$

## 核心要点
1. RTF < 1 是流式语音交互的必要条件
2. 越低越好: RTF 越小，系统越有余量处理并发请求
3. Qwen3-Omni RTF 范围 0.47-0.66（单并发至高并发）

## 代表工作
- [[Qwen3-Omni]]: 多并发下 RTF 始终 < 1

## 相关概念
- [[First-Packet Latency]]
- [[Streaming Speech Synthesis]]
