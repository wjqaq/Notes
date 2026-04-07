CVPR 2025

传统的DPO是离线[Offline RL（离线强化学习）](../相关概念.md#Offline%20RL（离线强化学习）)异策略[Off-Policy（异策略）](../相关概念.md#Off-Policy（异策略）)，其行为策略和目标策略的数据集分布往往是偏离的。
#### 问题
![](assets/OPA-DPO（On-Policy%20Alignment%20DPO）/file-20260403170657832.png)

- 图a左，黑线表示$\pi_{ref}$ 参考模型，红色的圈（幻觉的错误回答）大都落在黑色内部，可以看到模型本身就倾向于生成错误的答案；绿色正方形（无幻觉的回复）在横轴右侧，里黑线很远，意思是这些正确回复对于初始模型来看，生成几率几乎为0；由于DPO限制机制（KL散度）它不允许模型在更新时偏离参考模型太远，尽管我们期望得到红线的位置，但最终也只能得到绿色的曲线，这就是传统DPO没有办法有效学习“异策略”的偏好回复；
- 图a右，OPA（同策略对其），本质上是先用优质数据对模型做一次基本的监督微调（SFT），然后就将黑线模型$\pi_{ref}$拉成了橙色的曲线$\pi_{OPA}$；这样原本离参考模型很远的正确答案，被覆盖到了$\pi_{OPA}$的范围内了，变成了同策略数据；在用$\pi_{OPA}$为起点跑DPO算法，这样就没有巨大的KL散度阻力了，最终就可以训练出绿色的$\pi_{OPA-DPO}$了；
- 图b，说明了OPA-DPO用少量数据达到最低幻觉率，说明，把数据变成“同策略”后，模型的学习效率会大幅提升；
- 图c：只做DPO或OPA效果提升没OPA-DPO要好。
#### 思考
##### Q1：相对于初始/参考策略的数据集分布如何影响DPO性能？
定义：
- $\mathcal{Y}_{global}$：**全局采样空间**。代表语言模型理论上能组合出的所有可能的文本回复（无论好坏、无论逻辑通不通）。
- $\mathcal{Y}_{ref}$：**参考测量支持集**。模型在初始状态下（即 $\pi_{ref}$）有能力生成的回复，换句话说，就是初始模型生成概率大于 $0$ 的回复集合。
- $\mathcal{Y}_{\theta}$：**更新策略支持集**。我们希望通过 DPO 训练后，模型 ($\pi_{\theta}$) 能够生成的回复集合。
![](assets/OPA-DPO（On-Policy%20Alignment%20DPO）/file-20260407130359577.png)
$$\mathbb{D}_{KL}[P||Q]:=\sum_{y\in\mathcal{Y}}P(y)\log\frac{P(y)}{Q(y)}$$
给定一个提示 $x$ 和图像 $m$，假设存在一个响应 $y$ 使得 $\pi_{\theta}(y|x,m)>0$，而 $\pi_{ref}(y|x,m)\rightarrow 0$，则两个策略之间的KL散度将变为 $\mathbb{D}_{KL}[\pi_{\theta}(\cdot|x,m)||\pi_{ref}(\cdot|x,m)]\rightarrow \infty$。
- 由图a左可以看出如果专家给出了一个完美的回复 $y$，但这个回复落在了 $\mathcal{Y}_{global} \setminus \mathcal{Y}_{ref}$ 中。这意味着在初始模型的认知里，这句话太陌生了，生成它的概率极低，即 $\pi_{ref}(y) \rightarrow 0$ ；
- 此时，如果我们强行要求更新后的模型去学习并输出这句话，即要求 $\pi_{\theta}(y) > 0$ ；
- 在代入 KL 散度公式时，分母 $\pi_{ref}(y)$ 趋近于 $0$，而分子 $\pi_{\theta}(y)$ 大于 $0$。这会导致对数内部的除法结果趋近于无穷大，最终使得整个 KL 散度惩罚 $\mathbb{D}_{KL} \rightarrow \infty$ 。
由此推导出：
$\mathcal{Y}_{\theta} \subseteq \mathcal{Y}_{ref} \subseteq \mathcal{Y}_{global}$ 。**只要是初始模型 $\pi_{ref}$ 绝对说不出的话，无论答案多好，更新后的模型 $\pi_{\theta}$ 也绝对说不出。**

##### Q2：其他采用DPO解决幻觉问题的算法有哪些内在缺陷？

![](assets/OPA-DPO（On-Policy%20Alignment%20DPO）/file-20260407102808518.png)
DPO解决LVLM幻觉问题分为三类：
- 幻觉注入：直接拿一个绝对正确的人工标注答案（Ground-truth）作为“好回复”，然后人为地往里面“注入”一些幻觉（比如故意改错几个物体名字或者数量），做成“坏回复” 。把这对数据喂给 DPO 进行训练。代表算法有 POVID 和 HALVA ；
	- 缺点：这些幻觉是人故意捏造出来的，根本**不是模型自己固有的错误** 。虽然模型知道了正确答案长什么样，但它自己真正容易犯的那些“典型幻觉”并没有被揪出来，自然也就无法得到针对性的纠正 。
- 幻觉识别：先让模型自己看着图片生成一段回复，然后请一位“专家”（比如功能更强大的 GPT-4V 或人类标注员）来挑错，并把这段回复修改成完美的“好回复” 代表算法有 RLHF-V, HA-DPO 等 ；
	- 缺点：专家修改后的完美答案，往往带有专家自己的逻辑和词汇，对原始模型来说太陌生了（在模型看来，生成这种完美答案的概率极低）。正如前面图1分析的，传统的 DPO 算法很难把这种“超纲”的优质答案学进去 。
- 自我进化：让模型自己对同一张图片生成多个不同的回复 。然后找个“裁判”（通常是更强的AI）来对比一下，哪个回复的幻觉稍微少一点，哪个就是“好回复” 。代表算法是 RLAIF-V ；
	- 缺点：**两个回复都是这个原本就有幻觉倾向的模型生成的**，很可能两个回复里都依然存在严重的幻觉，拿着都不干净的答案去训练，学习效率非常低，往往需要反复迭代并堆积海量的数据才能看到一点效果 。

##### Q3：可以对当前框架进行哪些调整以纠正其固有的缺陷？
方法3的操作方式是：让**当前正在训练的模型自己**对同一张图片生成两个不同的回答，然后找个更强的AI裁判来评判哪个更好。**问题在于：** 这个模型本身就是有幻觉缺陷的。

因此提出OPA-DPO。
#### 方法
![](assets/OPA-DPO（On-Policy%20Alignment%20DPO）/file-20260407164710077.png)
数据集构建：
1. 将Prompt + images，给原始模型生成$y_{Gen}$；
2. 将Prompt + images + $y_{Gen}$+$y_{GT}$（正确答案）给GPT-4V，让专家去修改有幻觉的回复，得到$y_{Rev}$，但$y_{Rev}$是带有专家风格的；
	专家做的事：
	- 响应中的每个句子都被分配一个分数 $S_{hal}$，指示幻觉的严重程度。此外，要求GPT-4V将包含错误描述的句子分类为图像识别错误或语言理解错误，分类结果由 $S_{img}$ 表示；
	- 指示GPT-4V对任何错误的句子进行最小程度的修改，这些修改后句子的总和表示为 $y_{Rev}$。
3. 在正式做 DPO 之前，先用正确答案（$y_{GT}$）和修改后的答案（$y_{Rev}$）对初始模型进行一次简单的 LoRA-SFT（有监督微调），此时，原本属于“离策略”的优质答案，全部变成了模型自己也能顺畅说出来的“同策略 (On-policy)”数据。
4. 构成了3个维度（6个偏好对）数据集：
	- 语言纠正（Language Correction）：
		正确答案 ($y_{GT}$) > 初始生成答案 ($y_{Gen}$)；
		修改后答案 ($y_{Rev}$) > 初始生成答案 ($y_{Gen}$)；
	- 图像聚焦（Image Focus）：
		看着“清晰原图”生成的答案 > 看着“被马赛克破坏的图”生成的同款答案。
	- 锚定偏好（Anchor）：
		约束首选回复（$y_{GT}$ 和 $y_{Rev}$）的概率下降。
DPO训练：
- 语言纠正（Language Corrections）：
	构建了从GPT-4V标记的幻觉分数的映射，以建立更新权重 $W_{hal}(S_{hal})$。然后，幻觉加权对数策略定义为 $\log \pi^{hw}(y|x,m)=\sum_{i}^{L}W_{hal}(S_{hal}^{i})\log \pi(y_{i}|x,m,y_{<i})$。
	以此建立语言修正偏好对：
	$$\mathcal{L}_{LC}=-\mathbb{E}_{(y_{GT},y_{Rev},y_{Gen},x,m)\sim\mathcal{D}}[\log \sigma(\beta \log\frac{\pi_{\theta}(y_{GT}|x,m)}{\pi_{OPA}(y_{GT}|x,m)}-\beta \log\frac{\pi_{\theta}(y_{Gen}|x,m)}{\pi_{OPA}(y_{Gen}|x,m)}) + \log \sigma(\beta \log\frac{\pi_{\theta}^{hw}(y_{Rev}|x,m)}{\pi_{OPA}^{hw}(y_{Rev}|x,m)}-\beta \log\frac{\pi_{\theta}^{hw}(y_{Gen}|x,m)}{\pi_{OPA}^{hw}(y_{Gen}|x,m)})]$$
- 图像聚焦机制 (Image Focus Mechanism)：
	创建了从GPT-4V分类结果 $S_{img}$ 的映射来确定更新权重 $W_{img}(S_{img})$。图像加权对数策略被描述为 $\log \pi^{iw}(y|x,m)=\sum_{i}^{L}W_{img}(S_{img}^{i})\log \pi(y_{i}|x,m,y_{<i})$。
	以此建立图像聚焦偏好对：$$\mathcal{L}_{IF}=-\mathbb{E}_{(y_{GT},y_{Rev},x,m,m^{\prime})\sim\mathcal{D}}[\log \sigma(\beta \log\frac{\pi_{\theta}(y_{GT}|x,m)}{\pi_{OPA}(y_{GT}|x,m)}-\beta \log\frac{\pi_{\theta}(y_{GT}|x,m^{\prime})}{\pi_{OPA}(y_{GT}|x,m^{\prime})}) + \log \sigma(\beta \log\frac{\pi_{\theta}^{iw}(y_{Rev}|x,m)}{\pi_{OPA}^{iw}(y_{Rev}|x,m)} - \beta \log\frac{\pi_{\theta}^{iw}(y_{Rev}|x,m^{\prime})}{\pi_{OPA}^{iw}(y_{Rev}|x,m^{\prime})})]$$
- 锚定偏好（Anchored Preference)：
	采用两个锚点来约束首选响应的概率不至于下降太多：
	$$\mathcal{L}_{Anc}=-\mathbb{E}_{(y_{GT},y_{Rev},x,m)\sim\mathcal{D}}[\log \sigma(\beta \log\frac{\pi_{\theta}(y_{GT}|x,m)}{\pi_{OPA}(y_{GT}|x,m)}-\delta) + \log \sigma(\beta \log\frac{\pi_{\theta}(y_{Rev}|x,m)}{\pi_{OPA}(y_{Rev}|x,m)}-\delta)]$$
结合以上公式，我们得到OPA-DPO的最终损失函数：

$$\mathcal{L}_{OPA-DPO}=\mathcal{L}_{LC}+\gamma_{1}\mathcal{L}_{IF}+\gamma_{2}\mathcal{L}_{Anc}$$

#### 实验
![](assets/OPA-DPO（On-Policy%20Alignment%20DPO）/file-20260407205550070.png)
为了证明离策略的首选响应无法通过DPO有效学习，我们将不同模型在200个显著修改的响应上的Token平均对数概率进行了可视化。在没有OPA的情况下进行DPO训练后，分布显示出可忽略不计的变化；而在我们的OPA-DPO下观察到了显著增加。
##### 结果
![](assets/OPA-DPO（On-Policy%20Alignment%20DPO）/file-20260407205819396.png)
![](assets/OPA-DPO（On-Policy%20Alignment%20DPO）/file-20260407210005212.png)
可以看出仅训练600条数据，幻觉指标也超过了大部分基准模型。

##### 消融
