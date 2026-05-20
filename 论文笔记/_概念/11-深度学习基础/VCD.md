---
type: concept
aliases: [Visual Contrastive Decoding, 视觉对比解码]
---

# VCD

## 定义
Visual Contrastive Decoding，通过对比原始图像和加噪图像下的 logit 分布差异，放大视觉条件信号以抑制幻觉的推理时方法。

## 核心要点
1. 在解码时同时前向原始图像和加噪版本，利用两者的 logit 差异识别视觉敏感 token
2. 训练无关，即插即用
3. 仅在视觉模态有效，无法扩展到音频
4. [[LIME]] 中的主要 baseline 之一

## 代表工作
- (Leng et al., 2024): VCD 原始提出
- [[MHSA]]: 与 VCD 对比——MHSA 仅在检测到幻觉时额外计算（+0.43x），VCD 固定双倍前向（+1x）
- [[FLB]]: 指出 VCD 存在[[长程衰减]]问题，且需双倍推理时间

## 相关概念
- [[多模态幻觉]]
- [[ICD]]
- [[LIME]]
- [[MHSA]]
- [[FLB]]
- [[长程衰减]]
