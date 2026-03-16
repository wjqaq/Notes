#### ⌚背景
- **NLP 领域的 Transformer 范式成熟**：Transformer 早已成为 NLP 任务的标配，基于 “大规模语料预训练 + 下游任务微调” 的范式，实现了极致的模型缩放与性能提升，千亿参数模型仍未出现性能饱和；
- **CV 领域的 CNN 主导困境**：当时 CV 领域仍以 CNN 为绝对主流，过往尝试将自注意力应用于视觉的工作，要么将注意力与 CNN 结合，要么仅替换 CNN 的部分组件，要么使用特殊的局部注意力模式，无法实现硬件友好的大规模缩放，经典 ResNet 类架构仍是大规模图像识别的 SOTA；
- 卷积神经网络具有两个归纳偏置（**Inductive Bias**，局部性：相邻图片具有相似特征，和平移不变性：无论是先做平移或者先做卷积得到的结果是不变的），因此卷积神经网络自带这样的先验信息，而Transformer需要自己去学习视觉信息，但CV任务是否必须依赖这些归纳偏置？
#### 🤖结构
![](assets/ViT(Vision%20Transformer)/file-20260316134223045.png)/file-20260316134223045.png)
$$
z_{0}=\left[x_{class } ; x_{p}^{1} E ; x_{p}^{2} E ; \cdots ; x_{p}^{N} E\right]+E_{pos }, E \in \mathbb{R}^{\left(P^{2} \cdot C\right) × D}, E_{pos } \in \mathbb{R}^{(N+1) × D} (1)
$$
$$
z_{\ell}'=MSA\left(LN\left(z_{\ell-1}\right)\right)+z_{\ell-1}, \ell=1 ... L (2)
$$
$$
z_{\ell}=MLP\left(LN\left(z_{\ell}'\right)\right)+z_{\ell}', \ell=1 ... L (3)
$$
$$
y=LN(z_{L}^{0}) (4)
$$
- 将图像分块投影后 + [class]token 后 加上位置编码信息作为Transformer Encoder输入；$E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 线性投影层的维度
- Encoder复用Transformer的编码器层，MSA为多头注意力机制，MLP是放大缩小（通常放大4倍）这里Transformer 输出维度  $\mathbb{R}^{(1)\times D}$将[class]作为整体特征输出。

- 