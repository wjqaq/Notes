
#### ⌚偏好对齐
##### 🤺RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）[ALIGNING LARGE MULTIMODAL MODELS  WITH FACTUALLY AUGMENTED RLHF](../RLHF/ALIGNING%20LARGE%20MULTIMODAL%20MODELS%20%20WITH%20FACTUALLY%20AUGMENTED%20RLHF.md)
###### 1. 有监督微调（SFT）
先用高质量的图文指令标注数据，微调基座 LVLM，让模型先学会基础的图文指令遵循、图像描述等能力，得到一个符合基础要求的 SFT 模型，作为后续强化学习的「初始策略模型」。
###### 2.训练独立的奖励模型
