# 关键代码


## CSR的数据集生成(用vllm部署模型 + openai 库调用)
sample.py
```python
from utils import *
import torch
from vllm import LLM, SamplingParams
from PIL import Image
import os
import argparse
import json
import time
import warnings
import logging

# 关闭所有冗余日志/警告
warnings.filterwarnings("ignore")
logging.getLogger("vllm").setLevel(logging.ERROR)
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

DEFAULT_IMAGE_TOKEN = "<image>"

def get_prompts(inputs):
    input_questions = [DEFAULT_IMAGE_TOKEN + '\n' + input_question for input_question in inputs]
    prompts = []
    for input_q in input_questions:
        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], input_q)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())
    return prompts

def calculate_sequence_logprob(completion):
    total_logprob = 0.0
    for token_logprob_dict in completion.logprobs:
        if token_logprob_dict:
            logprob_obj = next(iter(token_logprob_dict.values()))
            try:
                total_logprob += logprob_obj.logprob
            except AttributeError:
                total_logprob += float(logprob_obj)
    return total_logprob

# ====================== Beam 树核心逻辑 ======================
def sentence_level_beam_search_tree(qid, llm, tokenizer, initial_text, image, sentence_end_id, max_length, max_new_tokens, num_beams, num_beam_group, token_level_beams, diversity_penalty):
    root = Node(initial_text, 0, 0)
    active_nodes = [root]

    while active_nodes:
        new_nodes = []
        for node in active_nodes:
            current_prompt = node.text
            multimodal_prompt = {
                "prompt": current_prompt,
                "multi_modal_data": {"image": image}
            }
            sampling_params = SamplingParams(
                n=token_level_beams,
                stop_token_ids=[sentence_end_id],
                max_tokens=max_new_tokens,
                logprobs=1,
                temperature=0.5
            )
            outputs = llm.generate(prompts=multimodal_prompt, sampling_params=sampling_params)

            for output in outputs:
                for completion in output.outputs:
                    gen_text = completion.text
                    full_text = node.text + gen_text
                    total_logprob = calculate_sequence_logprob(completion)
                    new_score = node.score + total_logprob
                    is_final = (sentence_end_id in completion.token_ids) or (len(full_text) >= max_length)
                    new_node = Node(full_text, new_score, node.depth + 1, node, is_final)
                    node.add_child(new_node)
                    if not is_final:
                        new_nodes.append(new_node)

        new_nodes.sort(key=lambda x: x.score, reverse=True)
        active_nodes = new_nodes[:int(num_beams/2)-1] + new_nodes[-int(num_beams/2):] if len(new_nodes) >= num_beams else new_nodes
        if not active_nodes:
            break
    return [{'id': qid, 'tree': root}]

def eval_model(args):
    # ====================== 1. 加载数据集 ======================
    with open(args.dataset_path, 'r', encoding='utf8') as fp:
        dataset = json.load(fp)
    total = len(dataset)
    print(f"✅ 成功加载数据集：{total} 条样本")

    # ====================== 2. vLLM 初始化 ======================
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,  # 降低一点，避免显存崩溃
        max_model_len=2048,
        max_num_seqs=8,
        disable_log_stats=True,
        tensor_parallel_size=1,
    )
    tokenizer = llm.get_tokenizer()
    sentence_end_id = args.period_id

    # ====================== 3. 路径/参数初始化 ======================
    output_dir = args.output_dir
    images_dir = args.images_dir
    os.makedirs(output_dir, exist_ok=True)
    log_path = "progress.log"
    start_time = time.time()

    # ====================== 4. 逐样本处理 ======================
    for idx, data in enumerate(dataset):
        try:
            qid = idx + 1  
            # 对齐原版：清洗Prompt
            input_question = data['input'].replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "").strip()
            # 对齐原版：图片路径拼接
            image_name = data['image']
            image_path = os.path.join(images_dir, f'COCO_train2014_{image_name}')
            
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            # 生成对话Prompt
            prompts = get_prompts([input_question])
            
            # 执行 Beam 搜索
            result = sentence_level_beam_search_tree(
                qid, llm, tokenizer, prompts[0], image,
                sentence_end_id, args.max_length, args.max_new_tokens,
                args.num_beams, args.num_beam_group, args.num_token_beams, args.diversity_penalty
            )

            # 保存结果
            for obj in result:
                save_path = os.path.join(output_dir, f"{obj['id']}.pkl")
                save_object(obj, save_path)

            # 进度日志
            current = idx + 1
            progress = current / total * 100
            elapsed = time.time() - start_time
            eta = (elapsed / current) * (total - current) if current > 0 else 0
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"🚀 总进度: {current}/{total} | {progress:.2f}% | 已用时: {elapsed:.0f}s | 预计剩余: {eta:.0f}s\n")

        except Exception as e:
            print(f"❌ 样本 {idx+1} 处理失败：{str(e)}")
            continue

    # 完成标记
    with open(log_path, "a") as f:
        f.write("✅ ALL TASKS COMPLETED\n")
    print("✅ 所有任务处理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./models/llava-1.5-7b-hf')
    parser.add_argument("--dataset_path", type=str, default='./data/CSR-Prompt-Dataset-12k.json')
    parser.add_argument("--images_dir", type=str, default="./data/images/")
    parser.add_argument("--output_dir", type=str, default="./outputs/sample_vllm")
    parser.add_argument("--diversity_penalty", type=float, default=3.0)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_beam_group", type=int, default=5)
    parser.add_argument("--num_token_beams", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=70)
    parser.add_argument("--period_id", type=int, default=29889)
    args = parser.parse_args()

    eval_model(args)
```
执行:
```bash

```

## Qwen2-VL 的 Visual Contrastive Decoding(VCD)


# 成功运行截图
##### sample.py
🚀 总进度: 448/12856 | 3.48% | 已用时: 299s | 预计剩余: 8280s


