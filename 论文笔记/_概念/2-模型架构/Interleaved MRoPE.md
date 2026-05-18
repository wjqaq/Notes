---
type: concept
aliases: [交错MRoPE, Interleaved Multimodal RoPE]
---

# Interleaved MRoPE

## 定义
一种改进的多模态旋转位置编码方法，通过将时间(t)、水平(h)、垂直(w)三个维度在嵌入空间中交错分布，解决原始 [[MRoPE]] 的频率谱不平衡问题。

## 数学形式
传统 MRoPE 将 $d$-维嵌入划分为三个连续子空间分别编码 t/h/w 频率；Interleaved MRoPE 将 t/h/w 在嵌入维度上交错排列：

$$
\theta_{i} = \theta_{\text{base}}^{-2i/d}, \quad \text{其中 } i \in \{0,1,\dots,d/3-1\} \text{ 索引顺序为 } t,h,w,t,h,w,\dots
$$

## 核心要点
1. 解决 MRoPE 中 t/h/w 分块导致的频谱不平衡问题
2. 每个时空轴均匀覆盖低频和高频带，提升长距离位置建模
3. 显著改善长视频理解任务表现

## 代表工作
- [[Qwen3-VL]]: 首次提出并应用 Interleaved MRoPE

## 相关概念
- [[MRoPE]]
- [[Qwen2.5-VL]]
