---
type: concept
aliases: [bounding box, bbox, 边界框]
---

# Bounding Box

## 定义
计算机视觉中表示目标区域的最小外接矩形，由左上角和右下角坐标定义。

## 数学形式
$$B = (x_{\text{topleft}}, y_{\text{topleft}}), (x_{\text{bottomright}}, y_{\text{bottomright}})$$

## 核心要点
1. Qwen-VL 将 bounding box 坐标归一化到 [0, 1000) 并用文本字符串表示，实现 grounding 的文本化
2. 使用 `<box>` 和 `</box>` 包裹坐标字符串，`<ref>` 和 `</ref>` 标记被引用文本
3. 这种设计使 LLM 无需额外 vocabulary 即可处理定位任务

## 代表工作
- [[Qwen-VL]]: 用文本化 bbox 实现视觉定位
- [[Shikra]]: 类似方案

## 相关概念
- [[Visual Grounding]]
- [[Object Detection]]
