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
from vllm.logits_processors import LogitsProcessor
from transformers import AutoTokenizer

# ====================== 1. 原论文 DDPM 噪声生成函数======================
def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int = 999) -> torch.Tensor:
    """为图像添加DDPM扩散噪声，生成VCD对比用的失真图像"""
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
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
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, logits: torch.Tensor, ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """vLLM每步生成自动调用，校正logits"""
        # 获取噪声图像对应的模型输出
        noisy_logits = kwargs.get("noisy_image_logits")
        if noisy_logits is None:
            return logits


        vcd_logits = (1 + self.alpha) * logits - self.alpha * noisy_logits

        # === 自适应合理性约束 ===
        cutoff = torch.log(torch.tensor(self.beta, device=logits.device)) + logits.max(dim=-1, keepdim=True).values
        vcd_logits = vcd_logits.masked_fill(logits < cutoff, -float("inf"))

        return vcd_logits

# ====================== 3. 原POPE评估器======================
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

    def apply_vcd(self):
        """启用VCD解码"""
        self.sampling_params.logits_processors = [self.vcd_processor]

    def remove_vcd(self):
        """禁用VCD，恢复常规解码"""
        self.sampling_params.logits_processors = []

    def generate(self, prompt: str, image_path: str) -> str:
        """单图单轮推理，适配Qwen2-VL格式"""
        # 读取图像
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
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
                        help="COCO val2014数据集路径 (e.g., ./coco/val2014)")
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
            image_filename = f"{image_id:012d}.jpg"
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
            if (idx + 1) % 100 == 0:
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