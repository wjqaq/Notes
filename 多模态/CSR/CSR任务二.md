# 关键代码

## Qwen2-VL 的 Visual Contrastive Decoding(VCD)

```python 
import asyncio
import base64
import os
import json
from typing import List, Dict, Tuple
from PIL import Image
import torch
import clip
from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm.asyncio import tqdm

# ===================== 核心配置（对齐CSR论文） =====================
class CSRConfig:
    # 束搜索配置
    num_beams: int = 5  # 每一步的束数量
    max_sentences: int = 8  # 最大生成句子数
    max_new_tokens_per_sentence: int = 74  # 单句最大token数（对齐论文）
    sentence_separators: Tuple[str] = (".", "!", "?")  # 句子结束符
    # 奖励配置
    clip_model_name: str = "ViT-L/14"  # 和LLaVA/Qwen2-VL的视觉编码器对齐
    lambda_reward: float = 0.9  # 视觉奖励权重（论文默认0.9）
    # 模型配置
    model_name: str = "Qwen2-VL-7B-Instruct"  # 你的vLLM服务模型名
    vllm_base_url: str = "http://localhost:8000/v1"  # vLLM服务地址
    api_key: str = "dummy"
    # 输出配置
    output_dataset_path: str = "./csr_preference_data.json"
    max_concurrent: int = 10  # 最大并发处理数

# ===================== CLIP视觉奖励计算模块 =====================
class CLIPRewardCalculator:
    def __init__(self, config: CSRConfig, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(config.clip_model_name, device=device)
        self.lambda_reward = config.lambda_reward

    def calculate_image_sentence_similarity(self, image_path: str, sentence: str) -> float:
        """计算图像和句子的CLIP余弦相似度，对应论文的R_I(s)"""
        try:
            # 预处理图像
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            # 预处理文本
            text = clip.tokenize([sentence], truncate=True).to(self.device)
            # 计算特征
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)
                # 归一化
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # 余弦相似度
                similarity = (image_features @ text_features.T).item()
            # 对齐论文公式：max(100 * cos_sim, 0)
            return max(100 * similarity, 0.0)
        except Exception as e:
            print(f"CLIP similarity error: {str(e)}")
            return 0.0

    def calculate_calibrated_reward(
        self,
        image_path: str,
        sentence: str,
        sentence_logprob: float
    ) -> Tuple[float, float, float]:
        """计算校准奖励R(s)，返回R_I, R_T, R"""
        r_i = self.calculate_image_sentence_similarity(image_path, sentence)
        r_t = sentence_logprob  # 句子级累积对数概率（对应论文的R_T）
        r = self.lambda_reward * r_i + (1 - self.lambda_reward) * r_t
        return r_i, r_t, r

# ===================== 句子级束搜索核心逻辑 =====================
async def generate_candidate_sentences(
    client: AsyncOpenAI,
    config: CSRConfig,
    prompt: str,
    image_base64: str,
    prefix_text: str = ""
) -> List[Dict]:
    """单步生成候选句子，返回带对数概率的候选列表"""
    content = [
        {"type": "text", "text": prompt + prefix_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    ]

    try:
        response = await client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=config.max_new_tokens_per_sentence,
            temperature=0.7,
            top_p=0.9,
            n=config.num_beams,
            logprobs=True,
            top_logprobs=1,
            stop=list(config.sentence_separators),  # 句子结束符停止
            timeout=60
        )
    except Exception as e:
        print(f"Sentence generation error: {str(e)[:100]}")
        return []

    candidates = []
    for choice in response.choices:
        sentence = choice.message.content.strip()
        if not sentence:
            continue
        # 计算句子级累积对数概率
        total_logprob = 0.0
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                total_logprob += token_info.logprob
        # 补全句子结束符
        if choice.finish_reason == "stop":
            sentence += choice.stop_reason
        candidates.append({
            "sentence": sentence,
            "cumulative_logprob": total_logprob,
            "full_text": prefix_text + sentence
        })
    return candidates

async def generate_full_responses(
    client: AsyncOpenAI,
    reward_calculator: CLIPRewardCalculator,
    config: CSRConfig,
    prompt: str,
    image_path: str
) -> Tuple[str, str]:
    """完整的句子级束搜索，返回chosen和rejected响应"""
    # 编码图片
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # 初始化束：每个元素是 (full_text, cumulative_reward, sentence_list)
    beams = [{"full_text": "", "cumulative_reward": 0.0, "sentences": []}]
    final_candidates = []

    # 句子级束搜索迭代
    for step in range(config.max_sentences):
        new_beams = []
        # 并行处理当前所有束
        tasks = []
        for beam in beams:
            task = generate_candidate_sentences(
                client=client,
                config=config,
                prompt=prompt,
                image_base64=image_base64,
                prefix_text=beam["full_text"]
            )
            tasks.append((beam, task))

        # 执行生成
        for beam, task in tasks:
            candidates = await task
            for cand in candidates:
                # 计算当前句子的校准奖励
                r_i, r_t, r = reward_calculator.calculate_calibrated_reward(
                    image_path=image_path,
                    sentence=cand["sentence"],
                    sentence_logprob=cand["cumulative_logprob"]
                )
                # 更新累积奖励
                new_cumulative_reward = beam["cumulative_reward"] + r
                new_beam = {
                    "full_text": cand["full_text"],
                    "cumulative_reward": new_cumulative_reward,
                    "sentences": beam["sentences"] + [cand["sentence"]]
                }
                new_beams.append(new_beam)

        if not new_beams:
            break

        # 保留top-k和bottom-k的束，用于后续生成和最终偏好对
        new_beams_sorted = sorted(new_beams, key=lambda x: x["cumulative_reward"], reverse=True)
        top_k = new_beams_sorted[:config.num_beams]
        bottom_k = new_beams_sorted[-config.num_beams:]
        beams = top_k + bottom_k

        # 收集最终候选
        final_candidates.extend(new_beams)

    if not final_candidates:
        return "", ""

    # 排序选最优和最差，构造偏好对
    final_candidates_sorted = sorted(final_candidates, key=lambda x: x["cumulative_reward"], reverse=True)
    chosen_response = final_candidates_sorted[0]["full_text"].strip()
    rejected_response = final_candidates_sorted[-1]["full_text"].strip()
    return chosen_response, rejected_response

# ===================== 主流程：批量构造CSR数据集 =====================
async def main():
    config = CSRConfig()
    # 初始化客户端和奖励计算器
    client = AsyncOpenAI(base_url=config.vllm_base_url, api_key=config.api_key)
    reward_calculator = CLIPRewardCalculator(config)

    # 加载源数据集（论文用LLaVA-150k，可替换为自定义数据集）
    print("Loading source dataset...")
    dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")
    # 筛选描述/推理类样本，对齐论文
    dataset = dataset.filter(lambda x: "describe" in x["conversations"][0]["value"].lower() or "what is" in x["conversations"][0]["value"].lower())
    dataset = dataset.select(range(1000))  # 可调整样本数量

    # 并发控制
    semaphore = asyncio.Semaphore(config.max_concurrent)
    preference_dataset = []

    async def process_single_sample(sample: Dict):
        async with semaphore:
            # 解析prompt和图像ID（LLaVA数据集格式）
            prompt = sample["conversations"][0]["value"]
            image_id = sample["image"]
            # 替换为你的COCO图像目录
            image_path = f"./data/coco/train2017/COCO_train2017_{image_id:012d}.jpg"
            if not os.path.exists(image_path):
                return None

            # 生成偏好对
            chosen, rejected = await generate_full_responses(
                client=client,
                reward_calculator=reward_calculator,
                config=config,
                prompt=prompt,
                image_path=image_path
            )
            if not chosen or not rejected:
                return None

            # 构造LLaMA-Factory兼容的DPO格式
            return {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": chosen}
                ],
                "rejected_conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": rejected}
                ],
                "image": image_id,
                "chosen_reward": chosen,
                "rejected_reward": rejected
            }

    # 批量处理
    print(f"Start processing {len(dataset)} samples...")
    tasks = [process_single_sample(sample) for sample in dataset]
    results = await tqdm.gather(*tasks, desc="Generating CSR preference data")

    # 整理结果
    for res in results:
        if res:
            preference_dataset.append(res)

    # 保存数据集
    with open(config.output_dataset_path, "w", encoding="utf-8") as f:
        json.dump(preference_dataset, f, ensure_ascii=False, indent=2)

    print(f"CSR dataset construction complete! Total valid samples: {len(preference_dataset)}")
    print(f"Dataset saved to: {config.output_dataset_path}")

if __name__ == "__main__":
    asyncio.run(main())
```
# 成功运行截图