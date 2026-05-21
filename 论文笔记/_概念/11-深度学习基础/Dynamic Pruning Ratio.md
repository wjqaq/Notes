---
type: concept
aliases: [动态剪枝率, adaptive pruning ratio, content-aware pruning]
---

# Dynamic Pruning Ratio

## 定义
根据输入图像的内容特征自适应确定 token 剪枝比例，而非使用全局固定的剪枝率，以在效率与精度间取得更好平衡。

## 数学形式

$$r_{\text{dyn}} = f_{\text{normalize}}(\phi)(1 - \rho)$$

其中 $\phi$ 为 token 间平均相似度，$\rho$ 为文本密度。

$$\phi = \frac{2}{N(N-1)} \sum_{i<j} \frac{\mathbf{T}_v^i \cdot \mathbf{T}_v^j}{\|\mathbf{T}_v^i\|_2 \cdot \|\mathbf{T}_v^j\|_2}$$

$$\rho_k = \frac{1}{h \times w} \sum_{i,j} \mathbb{I}(G(i,j) \geq \tau)$$

## 核心要点
1. 特征冗余度高（φ 大）→ 可剪更多；文本密度高（ρ 大）→ 需保留更多
2. 手写识别等低信息密度场景允许更高剪枝率（r≈0.30），文档提取需保守（r≈0.17）
3. 相比固定剪枝率，同平均剪枝率下可提升最高 13.5% 精度
4. 完全训练无关，仅依赖前向传播的中间特征

## 代表工作
- [[RTPrune]]: 首次在 OCR token 剪枝中引入动态剪枝率，联合 φ 和 ρ

## 相关概念
- [[Token Pruning]]
- [[Sobel Operator]]
- [[ℓ2-Norm Feature Selection]]
