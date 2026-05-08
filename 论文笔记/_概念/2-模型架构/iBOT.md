---
type: concept
aliases: [Image BERT Pre-training with Online Tokenizer, 遮蔽图像建模]
---

# iBOT

## 定义
Image BERT Pre-training with Online Tokenizer，结合 DINO 自蒸馏和 MIM 的视觉预训练方法，对遮挡 patch token 施加表示级损失。

## 数学形式
$$\mathcal{L}_{iBOT} = -\sum_{i=1}^{N} m_i \cdot h_t(f_t(I)_i)^T \log h_s(f_s(I_{mask})_i)$$

其中 $m_i \in \{0,1\}$ 为遮挡指示器，仅对遮挡 token 计算损失。

## 核心要点
1. 自蒸馏范式：学生处理遮挡图像，教师处理完整图像
2. 损失仅作用于被遮挡的 patch token（默认 75% mask ratio）
3. 与 DINO 共享教师网络，通过 EMA 更新
4. 作为 TIPS/TIPSv2 的 patch-level 训练组件

## 代表工作
- [[TIPSv2]]: 升级到 iBOT++，对可见 token 也计算损失
- [[TIPS]]: 将 iBOT 引入 VLP 训练
- Zhou et al. (ICLR 2022): 原始 iBOT 论文

## 相关概念
- [[iBOT++]]
- [[DINO]]
- [[Masked Image Modeling]]
- [[Vision Transformer]]
