---
type: concept
aliases: [Detecting Hallucinations by Cross-modal Attention Pattern]
---

# DHCP

## 定义
基于跨模态注意力模式的 LVLM 幻觉检测方法，通过训练轻量 MLP 判别器识别注意力中的幻觉模式。

## 核心要点
1. 提取 LVLM 输出 token 到视觉 token 的跨模态注意力作为输入特征
2. 训练二层 MLP（hidden=128）二分类幻觉/非幻觉
3. 不修改 LVLM 参数，仅在推理时监控注意力模式
4. 支持 sentence-level 检测（生成式任务平均注意力），判别式任务单 token 检测

## 代表工作
- [[DHCP]]: 提出跨模态注意力幻觉检测范式
- [[MHSA]]: 将 DHCP 判别器重用作 token-level 监督信号，扩展到幻觉抑制

## 相关概念
- [[Cross-Modal Attention]]
- [[多模态幻觉]]
- [[Hallucination Detector]]
