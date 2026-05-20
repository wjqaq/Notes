---
type: concept
aliases: [RefCOCO, RefCOCO+, RefCOCOg, 指称定位数据集]
---

# RefCOCO

## 定义
基于 MSCOCO 的指称定位 (Referring Expression Comprehension/Grounding) 数据集系列。

## 核心要点
1. 三个子数据集：
   - RefCOCO: 基于 MSCOCO，含 142K 指称表达
   - RefCOCO+: 禁止位置词（如 "left"）的指称表达，更强调外观描述
   - RefCOCOg: 更长的指称表达，包含更丰富的语义和常识
2. 两个核心任务：
   - REC (Referring Expression Comprehension): 根据文本描述定位目标边界框
   - RES (Referring Expression Segmentation): 根据文本描述分割目标掩码
3. 评测指标：IoU / gIoU / cIoU

## 代表工作
- [[MMGrounded-PostAlign]]: 在此基准上达到有竞争力的零样本定位性能
- LISA: 推理分割
- GLaMM: 像素级接地对话

## 相关概念
- [[视觉定位]]
- [[分割掩码]]
- [[边界框检测]]
