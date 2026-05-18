---
type: concept
aliases: [MHSA总训练目标, MHSA三目标联合优化]
---

# MHSA Training Objective

## 定义
MHSA 框架的三目标加权联合优化损失函数，对应幻觉抑制的三个设计 desiderata：修正注意力至非幻觉模式、最小化修正量、保持 LVLM 输出质量。

## 数学形式
$$
\mathcal{L}_{\text{total}} = \lambda_{\text{dg}} \cdot \mathcal{L}_{\text{dg}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}} + \lambda_{\text{LVLM}} \cdot \mathcal{L}_{\text{LVLM}}
$$

其中：
- $\mathcal{L}_{\text{dg}} = -\log D_{0}(\mathbf{A} + G(\mathbf{A}))$：检测器引导损失
- $\mathcal{L}_{\text{reg}} = \|G(\mathbf{A})\|_{2}^{2}$：正则化损失
- $\mathcal{L}_{\text{LVLM}} = \text{CE}(f_{\text{LVLM}}(\mathbf{A}'), y_{\text{gt}})$：LVLM 输出质量损失

## 核心要点
1. 三目标互补：分别对应幻觉修正、语义保持、质量维持
2. 消融实验确认三者缺一不可（F1 从 92.97 降至 83-90 区间）
3. 生成式任务中 $\lambda_{\text{LVLM}}=0$（离线训练模式），因在线训练会导致过度生成抑制

## 代表工作
- [[MHSA]]: 提出该训练目标

## 相关概念
- [[Detector-Guided Loss]]
- [[Regularization Loss]]
- [[LVLM Output Quality Loss]]
- [[Adversarial Training]]
