---
type: concept
aliases: [TM-RoPE, Time-aligned Multimodal RoPE, M-RoPE, 多模态旋转位置编码]
---

# Multimodal Rotary Position Embedding (TM-RoPE)

## 定义
将 Rotary Position Embedding 分解为 temporal、height、width 三个维度，并引入绝对时间编码，实现多模态流中不同模态的精确时间对齐。

## 核心要点
1. 三个维度交错分配: temporal 24 角度、height 20、width 20（相比 M-RoPE 的 16 角度更平衡）
2. Text: 三维度共享相同位置 ID，等价于 1D RoPE
3. Audio: 共享位置 ID + 绝对时间编码，每 80ms 一个 temporal ID
4. Image: 恒定 temporal ID，独立 height/width ID
5. Video: 每帧单调递增 temporal ID，按实际时间戳动态调整至 80ms 分辨率
6. 多模态流位置编号连续，避免冲突

## 代表工作
- [[Qwen2.5-Omni]]: M-RoPE 首次提出
- [[Qwen3-Omni]]: 升级为 TM-RoPE，角度分配优化，直接按时间 ID 对齐

## 相关概念
- [[Rotary Position Embedding|RoPE]]
- [[Chunked Prefilling]]
