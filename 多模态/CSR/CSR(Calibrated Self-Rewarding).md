
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

##### 阶段二：偏好策划和微调阶段