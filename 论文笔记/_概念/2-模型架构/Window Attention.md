---
type: concept
aliases: [窗口注意力, 局部注意力]
---

# Window Attention

## 定义
把自注意力限制在局部窗口内的变体（代表 Swin Transformer）。

## 核心要点
1. 牺牲全局感受野换取计算效率 / 避免全局 shortcut
2. [[LaSt-ViT]] 用其作为消融基线验证'全局依赖'是惰性聚合的帮凶
3. 但单纯窗口化不能从根本解决 artifacts

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Vision Transformer]]
