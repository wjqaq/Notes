---
type: concept
aliases: [Bounding-box as Language Token, BTL训练]
---

# BTL

## 定义
Bounding-box as Language Token (BTL)，将边界框坐标编码为文本 token 进行统一训练的接地策略。

## 核心要点
1. 两种训练范式：
   - BTL-Generation: 图像+指称文本输入，MLLM 直接输出边界框坐标
   - BTL-Caption: 图像输入，输出包含边界框嵌入的文本描述
2. 局限：BTL-Generation 导致 MLLM 推理能力下降（过拟合视觉坐标信息）
3. 与显式视觉定位对比：BTL 缺乏像素级掩码，对齐强度不够
4. MMGrounded-PostAlign 的显式定位模块在所有指标上超越 BTL

## 代表工作
- [[MMGrounded-PostAlign]]: 证明 BTL 方法损害推理能力，显式定位模块更优
- Shikra: 早期 BTL 方法之一

## 相关概念
- [[边界框检测]]
- [[视觉定位]]
- [[多模态对齐]]
