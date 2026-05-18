---
type: concept
aliases: [LLaVA-v1.5, LLaVA-1.5]
---

# LLaVA

## 定义
LLaVA (Large Language and Vision Assistant) 是一个开源的大规模视觉语言模型系列，LLaVA-v1.5 是改进的基线版本，使用 7B/13B 参数。

## 核心要点
1. LLaVA-v1.5-7B 使用 32 层 Transformer (Vicuna-7B 骨干)
2. 视觉编码器使用 CLIP ViT-L
3. 是一种广泛使用的 LVLM 幻觉评估基准模型
4. 多模态融合呈现[[阶段化融合]]模式

## 代表工作
- [[SIRA]]: 使用 LLaVA-v1.5-7B 作为评估骨干，在 $b=16$ 处分叉最优
