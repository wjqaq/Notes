CVPR 2026


ViT 被用于全监督、文本监督的图像-文本对（CLIP模型）、自监督（仅图像作为训练的DINO模型）
#### ❔问题
![](assets/Vision%20Transformers%20Need%20More%20Than%20Registers/file-20260422102831060.png)
如，CLIP模型，拿"Cat"文本让模型去标出猫的位置，结果可能标注的边缘位置模糊，漏标严重
因为CLIP模型的训练是对齐<整张图片，文本>，模型发现只要背景“猫”出现的环境就能蒙对，就不需要去精准定位“猫”本身，导致密集特征和语义严重不对齐；
![](assets/Vision%20Transformers%20Need%20More%20Than%20Registers/file-20260421212454170.png)
而DINO，DINOv2模型的背景区域比前景还”亮“，出现高范数伪影，不加Register隔离无法使用。
这些问题都是ViT存在不同范式的问题：
- 检测定位不准
- 分割边缘模糊
- 开放词汇泛化崩溃
- 高范数伪影

为了在不同范式中建立这些现象的统一定义，我们引入了Patch Score：Patch features和CLS Token的相似度；
为了定量评估评分中的伪影，我们提出了Point-in-Box。
基于Patch Score 和 Point-in-Box，分析ViT的行为，并提出假设来解释伪影：
#### 假设
- 自然图像本质上包含许多与主要对象无关的背景 Patch。由于只有图像级别的监督，模型缺乏空间引导，因此倾向于通过背景证据来编码全局语义（即懒惰聚合，lazy aggregation）；
--实验，在预训练的 ViT 中移除得分最高的前 50% 的 Patch 对 ImageNet 准确率的影响微乎其微，这证实了这种依赖性。
- 全局依赖关系允许 ViT 利用这些无关的背景 Patch 作为表示全局语义的捷径。在缺乏 Patch 级别标注的情况下，ViTs 可能会在训练初期采取懒惰聚合策略，将微小的前景语义扩散到背景中。我们验证了减少全局依赖关系确实能缓解伪影现象。

具体分析
#### 分析
##### Patch Score
每个 Patch 与全局表示之间的相似度。
$$S_{p} = \frac{x_{patch} \cdot \mathcal{Q}_{CLS}}{\|x_{patch}\|_2 \|\mathcal{Q}_{CLS}\|_2} \quad (3)$$
其中：$$x_{patch}=\mathcal{P}_{enc}(\mathcal{P}_{emb}(x)), \quad \mathcal{Q}_{CLS}=Pooling(x_{patch}), \quad (1)$$
或：$$x_{patch}, \mathcal{Q}_{CLS} = \mathcal{P}_{enc}(\mathcal{P}_{emb}(x), \mathcal{O}_{CLS}), \quad (2)$$
##### Point-in-Box
基于 Patch 分数，我们通过判断得分最高区域是否对应于前景对象来评估伪影。我们使用 ImageNet 验证集中具有单一对象标注的图像来避免歧义。我们将框内点分数定义为最高 Patch 分数落在前景边界框内的图像比例。


CLS token 在“看”哪里？

![](assets/Vision%20Transformers%20Need%20More%20Than%20Registers/file-20260421220217107.png#pic_center)


**分布**：前景 Patch 集中在较低的 Patch 分数值处，而背景 Patch 则主导了高分尾部。
![](assets/Vision%20Transformers%20Need%20More%20Than%20Registers/file-20260421220413665.png#pic_center)
**掩蔽探测**：移除高分 Patch 不会损害准确率，甚至在超过 50% 的 Patch 被掩蔽时还能略微提高准确率（例如，ViT-B/16 提升 1.2%）。相比之下，移除低分 Patch 会导致准确率急剧下降。


这些现象何时开始？
![](assets/Vision%20Transformers%20Need%20More%20Than%20Registers/file-20260421221450474.png)
在 ImageNet-1k 上训练 ViT-B/16 和 ResNet-50：
- **框内点动态**：反映伪影水平的 ViT 框内点分数（越低表示背景偏见越强）在训练期间保持较低且几乎不变，即使在分类准确率提高的情况下也是如此。
- **与 ResNet 的比较**：与 ResNet 相比，ViT 始终显示出较低的框内点分数，尽管两者的图像级别准确率相似，但这揭示了更明显的背景偏见。
我们进一步假设这种行为源于两个相互作用的因素：(1) 粗粒度语义监督，图像级标签不能提供准确的 Patch 级监督；(2) 全局依赖关系，基于注意力的 token 混合允许背景 token 吸收前景信息。

##### 粗粒度语义监督：
![](assets/Vision%20Transformers%20Need%20More%20Than%20Registers/file-20260421222234893.png)
比较不同Patch 下的影响：
增大 Patch 尺寸后，框内点分数从 0.44 上升到 0.52。Patch 分数图显示高分区域从背景转移到了对象区域。然而，Top-1 准确率从 62% 降到了 55%，揭示了分类和定位准确率之间的权衡。

##### 懒惰行为源于 ViT 的全局依赖：
![](assets/Vision%20Transformers%20Need%20More%20Than%20Registers/file-20260421222352276.png)
不同层用基于窗口的注意力取代了全局自注意力：
随着全局注意力的受限，框内点分数增加，当所有层都采用窗口注意力时达到最高值。然而，准确率相应下降，这意味着全局上下文虽然有利于分类，但也促进了语义向背景 Patch 的扩散。





为了缓解懒惰聚合，提出方法
#### LaSt-ViT (LazyStrike ViT)
模型学会估计每个 token 的贡献，并选择性地将信息丰富的 Patch 特征整合到 CLS token 中，以强化前景表示。随着这些比例的适当增加，ViTs 会自动将其注意力转移到前景对象上，将高分 Patch 与前景对齐。

![](assets/Vision%20Transformers%20Need%20More%20Than%20Registers/file-20260421223541837.png)
因此筛选特征稳定的Patch是关键。
##### Stability Score（稳定性分数）
令 $x_{patch}\in\mathbb{R}^{N\times D}$ 表示由 ViT 编码器生成的所有 Patch 表示（在丢弃 [CLS] 之后），并令 $g\in[0,1]^{D}$ 为被复制到所有 Patch 的归一化高斯权重向量：
$$x_{FFT}=FFT1D(x_{patch})$$

$$x_{LP}=x_{FFT} \odot g$$

$$\hat{x}_{patch}=\mathfrak{R}\{IFFT1D(x_{LP})\} \quad (4)$$

$$S_{i,j} = \frac{\hat{x}_{patch}[i,j]}{x_{patch}[i,j] - \hat{x}_{patch}[i,j] + \epsilon} \quad (5)$$
对每个Patch做一维傅里叶变换，在经过高斯低通再逆变换，算出稳定分数。
##### Channel-wise Top-K Pooling（通道级 Top-K 池化）
利用通道级稳定性分数，我们通过在每个通道选择 $K$ 个最稳定的 Patch（token）并对它们求平均值，将 Patch 表示聚合到 CLS token 中：
$$\mathcal{I}_{K}(j)=TopK(\{S_{i,j}\}_{i=1}^{N},K), \quad j=1,...,D \quad (6)$$

$$\mathcal{Q}_{CLS}[j]=Pool_{K}(x_{patch}[:,j];S_{:,j}) \triangleq \frac{1}{K}\sum_{i\in\mathcal{I}_{K}(j)}x_{patch}[i,j], \quad j = 1,..., D \quad (7)$$
每个通道只选择最稳定的K个Patch进行聚合，强迫CLS只相信最稳定的K个前景

##### Vote Count（投票计数）
我们将 token (Patch) $i$ 的投票计数定义为：

$$v_{i}\triangleq\sum_{j=1}^{D}\mathbf{1}\{i\in\mathcal{I}_{K}(j)\}, \quad i=1,...,N \quad (8)$$

用投票机制提纯前景，统计每个Patch在多个通道里被选中的次数，次数越高权重越大。

#### 实验
**特征范数和 Patch 分数中伪影的消除**：实验结果证明，LazyStrike 不仅消除了高范数现象，还提高了框内点分数。应用 LazyStrike 后，ViT 的框内点分数接近 ResNet。对全监督训练下的特征范数进行的详细分析揭示，LazyStrike 降低了最大特征值，从而缓解了高范数现象。

| **方法**                | **高范数** | **框内点**      |
| --------------------- | ------- | ------------ |
| ResNet                | X       | 68.4         |
| ViT                   | ✓       | 42.7         |
| ViT (+LazyStrike)     | X       | 55.1 (+12.4) |
| DINO-ResNet           | X       | 71.1         |
| DINO-v1               | X       | 44.5         |
| DINO-v1 (+LazyStrike) | X       | 69.7 (+25.2) |
| CLIP-ResNet           | X       | 53.9         |
| CLIP                  | ✓       | 39.8         |
| CLIP (+LazyStrike)    | X       | 50.1 (+10.3) |
