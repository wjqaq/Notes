#### 模型所需显存

| 精度类型      | 单元素字节数 | 适用场景                  |
| --------- | ------ | --------------------- |
| FP32      | 4 字节   | 优化器主权重、梯度、优化器状态       |
| FP16/BF16 | 2 字节   | 模型推理、混合精度训练的前向 / 反向计算 |
| INT8      | 1 字节   | 量化推理 / 训练             |
| INT4      | 0.5 字节 | 量化推理 / QLoRA 训练       |
##### 推理所需显存

模型参数 + KV cache + 临时运行开销
- 模型参数显存：总参数 $\times$单参数字节数
	如：7B模型，FP16精度，参数显存 = $7 \times 10^9 \times 2$ = 14GB
- KV cache：是 Transformer 自注意力层的 key/value 向量缓存，用于生成新 token 时复用历史计算，是长文本推理的最大显存开销，计算公式：$batch\_size \times seq\_len \times num\_layers \times 2 \times 单元素字节$
	如：Llama2-7B（num_layers=32，hidden_size=4096），FP16 精度，batch_size=1，seq_len=4096
	KV cache 显存 = $1 \times 4096 \times 32 \times 2 \times 4096 \times 2$ = 2GB
- 临时运行开销：主要是向前传播的临时激活值、中间结果，占比低。

##### 训练所需显存
