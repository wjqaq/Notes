# 代码
#### 对比解码代码
$$
P_{vcd}(y|v,v',x) = softmax[(1+\alpha)logit_{\theta}(y|v,x) - \alpha logit_{\theta}(y|v',x)]
$$
$$
\mathcal{V}_{head}(y<_t) = \{ y_t \in \mathcal{V} : p_{\theta}(y_t|v,x,y<_t) \geq \beta  \max_{w}p_{\theta}(w|v,x,y<_t) \},
$$
$$
p_{vcd}(y_t|v,v',x) = 0 \; if \; y_t \notin \mathcal{V}_{head}(y<_t)
$$
源文件：
对比解码版本1：
```python
	probs = nn.functional.softmax(next_token_logits, dim=-1)
	cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values
	diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
	cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
```
对比解码版本2：
```python
	cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
	diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
	cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
```
评估代码pope_evaluation
```python
import torch
import torch.nn.functional as F
import asyncio
import os
import time
import base64
from typing import List, Dict
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessor,BatchUpdate
from transformers import AutoTokenizer

# ====================== 1. DDPM 噪声生成函数======================
def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int = 999) -> torch.Tensor:
    """为图像添加DDPM扩散噪声，生成VCD对比用的失真图像"""
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps, device=image_tensor.device)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    # 生成噪声并应用
    noise = torch.randn_like(image_tensor)
    noisy_img = alphas_bar_sqrt[noise_step] * image_tensor + one_minus_alphas_bar_sqrt[noise_step] * noise
    return noisy_img

# ====================== 2. vLLM VCD 对比解码 Logits 处理器 ======================
class VCDLogitsProcessor(LogitsProcessor):
    def __init__(
        self, 
        vllm_config: "VllmConfig",
        device: torch.device, 
        is_pin_memory: bool
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        self.alpha = 1.0
        self.beta = 0.1
        self.noisy_logits = None

    def set_noisy_logits(self, noisy_logits):
        self.noisy_logits = noisy_logits

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.noisy_logits is None:
            return logits
        
        # VCD 核心计算
        vcd_logits = (1 + self.alpha) * logits - self.alpha * self.noisy_logits
        cutoff = torch.log(torch.tensor(self.beta, device=logits.device)) + logits.max(dim=-1, keepdim=True).values
        vcd_logits = vcd_logits.masked_fill(logits < cutoff, -float("inf"))
        return vcd_logits

    def is_argmax_invariant(self) -> bool:
        return False
        
    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        pass

# ====================== 3. POPE评估器======================
class PopeEvaluator:
    """Class for Evaluating MLLMs on POPE dataset."""
    def __init__(self, *args, **kwargs):
        pass
        
    def evaluate(self, records: List[Dict]) -> Dict:
        metrics = {
            key: {
                "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                "count": 0, "accuracy": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
            }
            for key in ["random", "popular", "adversarial", "overall"]
        }
        
        for record in records:
            answer = record["answer"].strip().lower()
            response = record["response"].strip().lower()
            category = record["category"]
            
            ans_pos = 1 if answer == "yes" else 0
            neg_words = {"no", "not", "n't", "否", "不是", "没有"}
            resp_pos = 0 if any(word in response for word in neg_words) else 1
            
            metrics[category]["tp"] += ans_pos * resp_pos
            metrics[category]["tn"] += (1 - ans_pos) * (1 - resp_pos)
            metrics[category]["fp"] += (1 - ans_pos) * resp_pos
            metrics[category]["fn"] += ans_pos * (1 - resp_pos)
            metrics[category]["count"] += 1
            
            metrics["overall"]["tp"] += ans_pos * resp_pos
            metrics["overall"]["tn"] += (1 - ans_pos) * (1 - resp_pos)
            metrics["overall"]["fp"] += (1 - ans_pos) * resp_pos
            metrics["overall"]["fn"] += ans_pos * (1 - resp_pos)
            metrics["overall"]["count"] += 1
        
        for cat in metrics:
            count = metrics[cat]["count"]
            if count == 0:
                continue
            tp, tn, fp, fn = (metrics[cat]["tp"], metrics[cat]["tn"],
                             metrics[cat]["fp"], metrics[cat]["fn"])
            metrics[cat]["accuracy"] = (tp + tn) / count
            metrics[cat]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics[cat]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics[cat]["f1"] = 2 * metrics[cat]["precision"] * metrics[cat]["recall"] / (
                metrics[cat]["precision"] + metrics[cat]["recall"]
            ) if (metrics[cat]["precision"] + metrics[cat]["recall"]) > 0 else 0.0
        
        return metrics

# ====================== 4. Qwen2-VL vLLM 推理封装======================
class Qwen2VL_VCD:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        # 加载Qwen2-VL Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 加载vLLM Qwen2-VL模型
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1}
        )
        # 初始化VCD处理器
        self.vcd_processor = VCDLogitsProcessor(alpha=1.0, beta=0.1)
        # 固定采样参数（POPE评估要求确定性生成，temperature=0）
        self.sampling_params = SamplingParams(
            max_tokens=16,
            temperature=0.0,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        self.use_vcd = False

    def apply_vcd(self):
        """启用VCD解码"""
        self.use_vcd = True
        self.sampling_params.logits_processors = [self.vcd_processor]

    def remove_vcd(self):
        """禁用VCD，恢复常规解码"""
        self.use_vcd = False
        self.sampling_params.logits_processors = []

    def get_noisy_image_logits(self, prompt: str, image_base64: str):
        """生成噪声图像并获取模型logits（VCD核心逻辑）"""
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = {
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": f"data:image/jpeg;base64,{image_base64}"}
        }
        # 禁用VCD，仅获取噪声图像的原始logits
        old_use_vcd = self.use_vcd
        self.remove_vcd()
        # 推理获取logits（vLLM批量推理）
        outputs = self.llm.generate([inputs], self.sampling_params)
        # 恢复状态
        if old_use_vcd:
            self.apply_vcd()
        # 模拟噪声logits（实际项目中需替换为加噪声后的图像推理）
        return torch.randn((1, self.tokenizer.vocab_size), device="cuda")

    def generate(self, prompt: str, image_path: str) -> str:
        """单图单轮推理，适配Qwen2-VL格式"""
        # 读取图像
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        # ✅ 修复5：启用VCD时，设置噪声logits
        if self.use_vcd:
            noisy_logits = self.get_noisy_image_logits(prompt, image_base64)
            self.vcd_processor.set_noisy_logits(noisy_logits)
        
        # Qwen2-VL 官方指令模板
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # vLLM推理输入
        inputs = {
            "prompt": formatted_prompt,
            "multi_modal_data": {
                "image": f"data:image/jpeg;base64,{image_base64}"
            }
        }

        # 批量推理
        outputs = self.llm.generate([inputs], self.sampling_params)
        return outputs[0].outputs[0].text.strip()

# ====================== 5. 主评估流程 ======================
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2-VL-7B VCD POPE Evaluation")
    parser.add_argument("--coco-dir", type=str, required=True, 
                        help="COCO val2014数据集路径 (e.g., ./data/val2014)")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Qwen2-VL-7B 模型本地路径")
    parser.add_argument("--limit", type=int, default=None, 
                        help="测试用：限制评估样本数量")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, 
                        help="多卡并行数")
    args = parser.parse_args()

    # 1. 初始化模型
    print("="*50)
    print("初始化 Qwen2-VL-7B 模型...")
    model = Qwen2VL_VCD(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size
    )
    print("模型初始化完成！")
    print("="*50)

    # 2. 加载POPE数据集
    print("加载 POPE 数据集...")
    dataset = load_dataset("lmms-lab/POPE", split="test")   
    if args.limit:
        dataset = dataset.select(range(args.limit))
    print(f"总样本数: {len(dataset)}")

    # 3. 定义推理函数
    def run_inference(use_vcd: bool) -> List[Dict]:
        records = []
        if use_vcd:
            model.apply_vcd()
            print("\n===== 运行 VCD 解码模式 =====")
        else:
            model.remove_vcd()
            print("\n===== 运行 常规解码模式 =====")
        
        start_time = time.time()
        for idx, sample in enumerate(dataset):
            question = sample["question"]
            answer = sample["answer"]
            image_id = sample["image_id"]
            category = sample["category"]

            # 处理COCO图像路径（12位补零）
            image_filename = f"COCO_val2014_{image_id:012d}.jpg"  # ✅ 修复：val2014而非train2014
            image_path = os.path.join(args.coco_dir, image_filename)
            
            if not os.path.exists(image_path):
                print(f"缺失图像: {image_path}")
                continue

            # 推理
            try:
                response = model.generate(question, image_path)
            except Exception as e:
                print(f"推理失败: {str(e)}")
                response = ""

            # 保存记录
            records.append({
                "answer": answer,
                "response": response,
                "category": category
            })

            # 进度打印
            if (idx + 1) % 10 == 0:
                print(f"已处理: {idx+1}/{len(dataset)} | 耗时: {time.time()-start_time:.2f}s")

        total_time = time.time() - start_time
        print(f"推理完成 | 总耗时: {total_time:.2f}s")
        return records

    # 4. 执行两种模式推理
    print("\n开始评估...")
    # 常规解码
    regular_records = run_inference(use_vcd=False)
    # VCD解码
    vcd_records = run_inference(use_vcd=True)

    # 5. 评估指标计算
    evaluator = PopeEvaluator()
    regular_metrics = evaluator.evaluate(regular_records)
    vcd_metrics = evaluator.evaluate(vcd_records)

    # 6. 格式化输出结果
    print("\n" + "="*60)
    print("📊 常规解码 评估结果")
    print("="*60)
    for cat, metrics in regular_metrics.items():
        if metrics["count"] > 0:
            print(f"\n{cat.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")

    print("\n" + "="*60)
    print("🚀 VCD解码 评估结果")
    print("="*60)
    for cat, metrics in vcd_metrics.items():
        if metrics["count"] > 0:
            print(f"\n{cat.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")

    # 7. VCD效果提升对比
    print("\n" + "="*60)
    print("📈 VCD 效果提升对比 (F1 Score)")
    print("="*60)
    for cat in ["random", "popular", "adversarial", "overall"]:
        reg_f1 = regular_metrics[cat]["f1"]
        vcd_f1 = vcd_metrics[cat]["f1"]
        improve = vcd_f1 - reg_f1
        print(f"{cat.upper()}: {reg_f1:.4f} → {vcd_f1:.4f} (↑{improve:.4f})")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
```



```python
import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import set_seed
import warnings
warnings.filterwarnings("ignore")

# ===================== 官方规范导入 =====================
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor,
    BatchUpdate,
    MoveDirectionality
)

# ===================== 扩散加噪函数（不变） =====================
def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int) -> torch.Tensor:
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t].to(x_0.device)
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t].to(x_0.device)
        return alphas_t * x_0 + alphas_1_m_t * noise

    return q_x(image_tensor, noise_step)

# ===================== 🔥 官方规范 VCD LogitsProcessor =====================
class VCDLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        """官方强制要求的构造函数签名"""
        self.device = device
        self.req_config = {}  # key: batch_index, value: (alpha, beta)

    def is_argmax_invariant(self) -> bool:
        """VCD会改变最大值，必须返回False"""
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """官方强制：处理批次增删改"""
        if batch_update is None:
            return

        # 1. 移除结束的请求
        for idx in batch_update.removed:
            self.req_config.pop(idx, None)

        # 2. 添加新请求（从extra_args获取VCD参数）
        for idx, params, _, _ in batch_update.added:
            if params.extra_args and "cd_alpha" in params.extra_args:
                alpha = params.extra_args["cd_alpha"]
                beta = params.extra_args["cd_beta"]
                self.req_config[idx] = (alpha, beta)
            else:
                self.req_config.pop(idx, None)

        # 3. 处理移动/交换
        for src_idx, dst_idx, direction in batch_update.moved:
            src_val = self.req_config.pop(src_idx, None)
            dst_val = self.req_config.pop(dst_idx, None)

            if src_val is not None:
                self.req_config[dst_idx] = src_val
            if direction == MoveDirectionality.SWAP and dst_val is not None:
                self.req_config[src_idx] = dst_val

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """官方强制：批量处理logits，执行VCD"""
        if not self.req_config:
            return logits

        # batch格式：[源1, cd1, 源2, cd2, ...]
        src_indices = list(range(0, logits.shape[0], 2))  # 偶数位=源
        cd_indices = list(range(1, logits.shape[0], 2))   # 奇数位=对比

        # 只处理有VCD配置的成对请求
        for src_idx, cd_idx in zip(src_indices, cd_indices):
            if src_idx in self.req_config:
                alpha, beta = self.req_config[src_idx]

                # 你的原版VCD公式
                src_logits = logits[src_idx:src_idx+1]
                cd_logits = logits[cd_idx:cd_idx+1]

                cutoff = torch.log(torch.tensor(beta, device=src_logits.device)) + src_logits.max(dim=-1, keepdim=True).values
                corrected = (1 + alpha) * src_logits - alpha * cd_logits
                corrected = corrected.masked_fill(src_logits < cutoff, -float('inf'))

                # 写回源请求的logits
                logits[src_idx:src_idx+1] = corrected

        return logits

# ===================== POPE 评估函数（不变） =====================
def evaluate_pope(gt_path, pred_path):
    gt_data = [json.loads(line) for line in open(gt_path, "r", encoding="utf-8")]
    pred_data = [json.loads(line) for line in open(pred_path, "r", encoding="utf-8")]
    
    pred_map = {item["question_id"]: item["answer"] for item in pred_data}

    tp = tn = fp = fn = 0
    for gt in gt_data:
        qid = gt["question_id"]
        gt_label = gt["label"].lower().strip()
        pred_label = pred_map.get(qid, "").strip()

        if gt_label == "yes":
            tp += 1 if "yes" in pred_label else 0
            fn += 0 if "yes" in pred_label else 1
        else:
            tn += 1 if "no" in pred_label else 0
            fp += 0 if "no" in pred_label else 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "="*60)
    print("📊 POPE 评估结果 (Qwen2-VL + VCD + vLLM v1)")
    print("="*60)
    print(f"准确率 (Accuracy):\t{accuracy:.4f}")
    print(f"精确率 (Precision):\t{precision:.4f}")
    print(f"召回率 (Recall):   \t{recall:.4f}")
    print(f"F1分数 (F1 Score):\t{f1:.4f}")
    print(f"真阳性/真阴性/假阳性/假阴性: {tp}/{tn}/{fp}/{fn}")
    print("="*60)

# ===================== 主函数（官方规范调用） =====================
def main():
    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument("--model-path", default="./models/Qwen2-VL-7B-Instruct-AWQ")
    parser.add_argument("--image-folder", default="./data/val2014")
    parser.add_argument("--question-file", default="./data/POPE/coco_pope_popular.json")
    parser.add_argument("--answers-file", default="./output/pope_vcd_result.jsonl")
    # VCD
    parser.add_argument("--use-vcd", action="store_true", default=True)
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--noise-step", type=int, default=500)
    # 推理
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    # ===================== 🔥 官方规范：初始化时传入处理器 =====================
    llm = LLM(
        model=args.model_path,
        quantization="awq_marlin",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        enforce_eager=True,
        # ✅ 关键：在这里注册自定义LogitsProcessor
        logits_processors=[VCDLogitsProcessor] if args.use_vcd else [],
    )

    # ===================== 采样参数（通过extra_args传VCD参数） =====================
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        skip_special_tokens=True,
        logprobs=1,
        # ✅ 关键：用extra_args传递VCD参数
        extra_args={"cd_alpha": args.cd_alpha, "cd_beta": args.cd_beta} if args.use_vcd else {}
    )

    # 加载数据
    with open(args.question_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    results = []
    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch = questions[i:i+args.batch_size]
        requests = []

        for q in batch:
            img_path = os.path.join(args.image_folder, q["image"])
            prompt = f"<img>{img_path}</img>{q['text']} Answer with yes or no."

            # 1. 原图（源输入）
            img = Image.open(img_path).convert("RGB")
            requests.append({"prompt": prompt, "multi_modal_data": {"image": img}})

            # 2. 加噪图（对比输入）
            if args.use_vcd:
                img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
                noisy_tensor = add_diffusion_noise(img_tensor, args.noise_step)
                noisy_img = Image.fromarray((noisy_tensor.permute(1,2,0)*255).byte().numpy())
                requests.append({"prompt": prompt, "multi_modal_data": {"image": noisy_img}})

        # 推理
        outputs = llm.generate(requests, sampling_params, use_tqdm=False)

        # 提取结果（只取原图输出）
        for j, q in enumerate(batch):
            idx = j * 2 if args.use_vcd else j
            ans = outputs[idx].outputs[0].text.strip().lower()
            results.append({"question_id": q["question_id"], "answer": ans})

    # 保存
    with open(args.answers_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    evaluate_pope(args.question_file, args.answers_file)

if __name__ == "__main__":
    main()
```