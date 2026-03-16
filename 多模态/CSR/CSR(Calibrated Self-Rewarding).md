
#### ❓偏好对齐
##### 🤺RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）[ALIGNING LARGE MULTIMODAL MODELS  WITH FACTUALLY AUGMENTED RLHF](../RLHF/ALIGNING%20LARGE%20MULTIMODAL%20MODELS%20%20WITH%20FACTUALLY%20AUGMENTED%20RLHF.md)
##### 🤺DPO（Direct Preference Optimization）[DPO(Direct Preference Optimization)](../DPO/DPO(Direct%20Preference%20Optimization).md)


#### ⌚背景
- LVLM 通过指令调优融合预训练 LLM 与视觉模型，实现了强大的多模态能力，但普遍存在**幻觉现象**：生成的文本在语言学上合理，却与输入图像的视觉信息完全矛盾（比如描述图像中不存在的物体、属性）；
- 主流解决模态对齐的方案是**偏好优化**（如 RLHF、DPO），但存在两大无法回避的问题：
	- **资源成本极高**：依赖人工标注或 GPT-4 等外部大模型构建偏好数据，标注成本高、迭代效率低；
	- **偏好不匹配**：外部生成的偏好数据无法捕捉目标 LVLM 的内在偏好，目标模型能轻易区分人工构造的偏好对，导致优化效果大幅折扣。
- 利用纯文本的大模型响应生成和偏好建模显示出良好的对其效果，但是无法直接迁移到LVLM上。
#### 🤖流程
![](assets/CSR(Calibrated%20Self-Rewarding)/file-20260316171500622.png)/file-20260316171500622.png)
##### 阶段一：分步奖励建模与视觉校准
###### LVLM理想奖励准则：
- **视觉约束**：必须将图像 - 文本相关性融入奖励，解决模型忽略视觉输入的问题；
- **细粒度分步奖励**：不针对整段响应分配单一奖励，而是在生成的每一步（句子级别）分配奖励，提供更精准的引导，鲁棒性更强。
###### 句子级校准奖励机制
- **自生成指令遵循分数$R_T(s)$：**
$$
R_T(s) = \prod_{t=1}^{N_o}P(r_o|x,r_1,r_2,\cdots,r_{o-1})
$$
	其中$N_o$是句子s的token数量，$r_o$是句子中第$o$个token，该分数反应了生成文本遵循指令能力。
- **图像 - 响应相关性分数$R_I(s)$：**
	用于校准纯文本奖励的模态偏差，衡量生成句子与输入图像的视觉匹配度，采用与目标 LVLM 视觉编码器对齐的 CLIP 模型计算相似度：
$$
R_I(s) = max(100 * cos(\mathcal{F}_I(x_v),\mathcal{F}_T(s)),0)
$$
	其中，$\mathcal{F}_I(x_v)$是输入图像的CLIP视觉嵌入，$\mathcal{F}_T(s)$是生成句子的CILP文本嵌入，该分数越高，代表生成的句子与图像视觉内容越匹配。
- **最终校准奖励$R(s)$：**
$$
R(s) = \lambda \cdot R_I(s) + (1 - \lambda) \cdot R_T(s)
$$
整段响应$y$的累积奖励，是所有句子奖励之和：
$$R(y) = \sum_{i=1}^{N_y}R(s_i)$$
其中$N_y$是响应y的句子总数。
##### 阶段二：偏好策划和微调阶段
###### 句子级候选响应生成
![](assets/CSR(Calibrated%20Self-Rewarding)/file-20260316174713258.png)/file-20260316174713258.png)
采用句子级束搜索策略：
- 以句子结束符（如“."）为分隔，并行采样多个候选句子；
- 每个句子计算$R(s)$分数；
- 分别筛选出奖励最高、最低的 top-k 个句子，进入下一轮句子级束搜索；
- 循环上述过程，直到生成响应结束符eos，完成完整响应的生成；
- 计算每一条完整响应的累积校准奖励。
###### 偏好数据构建与 DPO 微调
