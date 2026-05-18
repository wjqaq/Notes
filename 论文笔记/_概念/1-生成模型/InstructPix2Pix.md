---
type: concept
aliases: [InstructPix2Pix, IP2P]
---

# InstructPix2Pix

## 定义
指令引导图像编辑的奠基性工作，将文本指令作为编辑信号直接嵌入扩散模型去噪过程，使模型能够根据自然语言指令编辑图像。

## 核心要点
1. 开创指令引导图像编辑范式：从掩码引导编辑转向自然语言指令编辑
2. 基于 Prompt-to-Prompt 和 Stable Diffusion，在生成的配对数据上训练
3. 模型能力有限，主要处理简单、显式的编辑命令

## 代表工作
- Brooks et al., "InstructPix2Pix: Learning to Follow Image Editing Instructions", CVPR 2023
- [[Entity-Rubrics]]: 作为奠基性工作引用

## 相关概念
- [[MagicBrush]]
- [[EMU-Edit]]
- [[Diffusion Models]]
