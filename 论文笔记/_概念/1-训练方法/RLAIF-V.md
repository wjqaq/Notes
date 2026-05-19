---
category: 1-训练方法
aliases: [RLAIF-V]
---

# RLAIF-V

基于开源 AI 反馈的 VLM 对齐方法，采用 Divide-and-Conquer 策略：将响应分解为子响应、聚合子响应评分得到总体评分，构建偏好数据用于 DPO 训练。使用 LLaVA-NeXT-34B 作为评判器，迭代式生成数据和重训练。

## 代表工作
- [[RLAIF-V]]: RLAIF-V (2024)
- [[TPR]]: 继承其分解+评分思路，改进为 topic 级选择性替换，效率更高
