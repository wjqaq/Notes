---
type: concept
aliases: [multi-granularity captions, 多粒度描述采样]
---

# Multi-Granularity Caption Sampling

## 定义
在视觉-语言预训练中交替使用不同粒度（PaliGemma 中等粒度 + Gemini Flash 高粒度）的合成图像描述，由 TIPSv2（CVPR 2026）提出。

## 核心要点
1. Web alt-text 信息缺失严重；PaliGemma 遗漏姿态、背景等细节
2. Gemini Flash 生成极详细描述，但过于全面会"轻视化"对比损失
3. 解决：随机交替使用 PaliGemma 和 Gemini 描述，保留对比学习难度
4. 配合双 CLS 设计：一个对 web alt-text，一个对合成描述
5. COCO T→I 检索 +4.4，DOCCI T→I +14.0

## 代表工作
- [[TIPSv2]]: 首次系统研究多粒度文本对 VLP 的影响

## 相关概念
- [[Contrastive Learning]]
- [[PaliGemma]]
- [[Gemini Flash]]
- [[InfoNCE Loss]]
