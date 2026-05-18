---
type: concept
aliases: [Rectified Linear Unit, 线性整流函数]
---

# ReLU

## 定义
线性整流函数（Rectified Linear Unit），定义为 $f(x) = \max(0, x)$，是深度学习中最常用的激活函数。

## 数学形式
$$
\text{ReLU}(x) = \max(0, x)
$$

## 核心要点
1. 计算简单：仅需比较和取最大操作，无指数运算
2. 缓解梯度消失：正区间梯度恒为 1（相比 Sigmoid/Tanh）
3. 稀疏激活：负输入输出为 0，带来隐式正则化
4. Dead ReLU 问题：负区间梯度为 0，某些神经元可能永久失活

## 代表工作
- [[MHSA]]: MLP 生成器和判别器均使用 ReLU 激活
- [[DHCP]]: 二层 MLP 检测器使用 ReLU

## 相关概念
- [[MLP]]
- [[GELU]]
