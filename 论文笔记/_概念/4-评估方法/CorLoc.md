---
type: concept
aliases: [correct localization, 正确定位率]
---

# CorLoc

## 定义
无监督物体发现的标准指标：预测框与 ground-truth 框的 IoU > 0.5 的图像比例。

## 核心要点
1. [[LaSt-ViT]] 用 [[Vote Count]] 阈值化得前景 mask 再取 bbox 评估 CorLoc
2. 在 VOC07/12 上超过 DINO + LOST 等专门方法
3. 常与 [[Unsupervised Object Discovery]] 搭配

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Unsupervised Object Discovery]]
- [[Vote Count]]
