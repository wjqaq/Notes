---
category: 1-训练方法
aliases: [Intra-Topic Self-Resampling]
---

# Intra-Topic Self-Resampling (Topic 内自重采样)

TPR 框架的核心组件之一。将 VLM 响应中的语义单元转换为 wh-question，然后使用参考模型自身在该 topic 下多次采样，为该 topic 生成多样化的替代候选语义单元。相比重采样整个响应，topic 级重采样避免了对所有 topic 同时要求正确的高难度，同时提供了选择性替换所需的细粒度候选池。

## 代表工作
- [[TPR]]: 作为 TPR 框架中 topic-level alternatives generation 的核心步骤
