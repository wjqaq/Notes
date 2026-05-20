---
type: concept
aliases: [Q-Former, Querying Transformer]
---

# Q-Former

## 定义
BLIP-2 中提出的跨模态对齐模块，使用可学习的 query token 通过交叉注意力从视觉编码器输出中提取与语言相关的视觉特征。

## 核心要点
1. 组件：包含一个 image transformer 和一个 text transformer，共享 self-attention 层
2. 用途：作为视觉编码器（如 CLIP ViT）和 LLM 之间的桥梁
3. 被 [[InstructBLIP]] 广泛使用——通过指令感知的 Q-Former 实现指令跟随的视觉特征提取
4. 对比 [[LLaVA]] 使用的简单 MLP projector，Q-Former 设计更复杂但提取能力更强

## 代表工作
- [[BLIP-2]]: Q-Former 原始提出
- [[InstructBLIP]]: 将 Q-Former 扩展为指令感知版本

## 相关概念
- [[InstructBLIP]]
- [[LLaVA]]
- [[CLIP]]
- [[大视觉语言模型]]
