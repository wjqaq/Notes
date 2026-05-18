---
type: concept
aliases: [注意力掩码, attention masking, image-token masking]
---

# Attention Mask

## 定义
Transformer 中控制 token 间注意力连接的机制，SIRA 利用它对反事实分支中的图像token位置施加后期阻断。

## 核心要点
1. 标准因果掩码 $M_t^{causal}$ 控制自回归生成的因果关系
2. SIRA 的反事实掩码 $M_t^{cf}$ 额外排除图像token作为 key 和 query
3. 在 prefill 阶段同时屏蔽图像token作为 query（防止其写入 [[KV Cache]]）
4. 在自回归解码阶段仅屏蔽图像token作为 key（新 token 不能回看图像）
5. 掩码在请求时固定，无需每次重建

## 代表工作
- [[SIRA]]: 通过反事实注意力掩码实现后期视觉阻断
