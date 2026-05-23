```python
对于判别任务：
1.询问：问用户“图里有沙发吗？”
2.定位：模型先不直接回答，而是用 grounding 能力强制输出它看到的“沙发”的框。
3.判断（逻辑校验）：
如果模型输出了坐标（比如 [100, 200, 300, 400]），说明它认为自己看到了沙发。
但是，检查框内的视觉语义（比如通过 CLIP 检查框内特征是否像沙发，或者使用另一个轻量级分类器确认）。
如果框内特征明显是“长凳”或“沙地”，则判定模型产生了幻觉，拒绝输出“有”。
4.修正回答：最终输出“没有沙发（检测模型误将长凳识别为沙发）”。

对于生成任务：
1.询问：问用户“这图片有什么？”
2.定位：让模型输出所有主要实体的边界框及其类别。比如在大象和游客那张图里，输出：
大象：[x1, y1, x2, y2]
游客A：[x3, y3, x4, y4]
桌子：[x5, y5, x6, y6]
长椅：[x7, y7, x8, y8]
属性验证：针对每个检测框，用简短语言描述其属性（由模型自身完成，或者用一个小模型（如CLIP）交叉验证）。
大象 -> 灰色，站在河边，有人骑在它身上。
桌子 -> 白色，四方形，放在游客旁边。
长椅 -> 木质，条纹状，放置在后排。
关系提取：基于框的位置信息，描述空间关系和动作。
大象在河边。
游客A（穿灰色背心）站在围栏旁看着大象。
桌子在游客A的前方。
3.组合生成：
当你拥有了实体清单、属性清单、关系清单三份准确数据后，再让模型将它们组合成连贯的语言。


针对这个设计提示词和整体架构。
```

```python
#!/usr/bin/env python3
"""
V9 GS: Logit-Space Bayesian Fusion
Qwen 存在性检测 → logit 空间融合 → LLaVA 最终决策

V7 核心问题:
  1. 二值门控: present AND boxes>0 缺一不可，浪费 Qwen 判断
  2. 对称惩罚: "不存在"惩罚与"存在"加权同力度 → FN↑
  3. 固定系数: 无自适应，|diff|=0.1 和 8.0 用同一个 0.5

V9 改进:
  1. tanh 连续映射: logit_diff → [-1,1] 软置信度
  2. 非对称设计: 负向 evidence 打折 50% → 保留 Recall
  3. 自动饱和: |Q|→∞ 时 tanh→±1，防止 Qwen 覆盖 LLaVA
  4. bbox 加分: 成功定位额外认可，但不强制要求
"""

import argparse, gc, io, json, math, os, re, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    LlavaForConditionalGeneration, LlavaProcessor,
    Qwen2_5_VLForConditionalGeneration, AutoProcessor,
)

PROJECT_ROOT = Path("/home/jay/Desktop/code/Project")
V9_OUTPUT = Path("/home/jay/Desktop/code/Project/v9/outputs")
LLAVA_PATH = str(PROJECT_ROOT / "models" / "llava-1.5-7b-hf")
QWEN_PATH = str(PROJECT_ROOT / "models" / "Qwen2.5-VL-3B-Instruct")
POPE_DIR = PROJECT_ROOT / "datas" / "POPE" / "Full"

ALPHA = 0.4
BETA_MAX = 3.0
GAMMA_NEG = 0.5
DELTA_BOX = 0.15


def load_pope(split, n=None):
    for pf in sorted(POPE_DIR.glob("*.parquet")):
        if split in pf.name:
            df = pd.read_parquet(pf)
            return df.sample(n=min(n, len(df)), random_state=42) if n else df
    return pd.DataFrame()


def extract_object(question):
    q = question.lower().rstrip("?").strip()
    pats = [
        r'(?:is|are) there (?:a |an |the |any )?(.+?) (?:in|on|at|near|shown|visible|on top of|sitting|standing|inside|within)',
        r'(?:is|are) there (?:a |an |the )?(.+?)\?',
        r'(?:is|are) there (?:a |an |the )?(.+?)$',
        r'(?:does|do) the (?:image|photo|picture) (?:contain|have|show|depict) (?:a |an |the |any )?(.+?)\?',
    ]
    for pat in pats:
        m = re.search(pat, q)
        if m:
            obj = m.group(1).strip().rstrip("?").rstrip(".").strip()
            obj = re.sub(r'^(a|an|the|some|any)\s+', '', obj)
            if 1 <= len(obj.split()) <= 5:
                return obj
    words = q.split()
    if words[:2] == ["is", "there"]:
        return " ".join(words[2:]).split("?")[0].strip()
    return q[:60]


def pil_from_parquet(img_data):
    if isinstance(img_data, dict):
        b = img_data.get("bytes")
    elif isinstance(img_data, bytes):
        b = img_data
    else:
        b = None
    if b is not None:
        return Image.open(io.BytesIO(b)).convert("RGB")
    return None


def metrics(df, preds):
    ys = df["answer"].str.lower().str.strip().values
    ps = [str(p).lower().strip() for p in preds]
    tp = sum(1 for y, p in zip(ys, ps) if y == "yes" and p == "yes")
    fp = sum(1 for y, p in zip(ys, ps) if y == "no" and p == "yes")
    fn = sum(1 for y, p in zip(ys, ps) if y == "yes" and p == "no")
    tn = sum(1 for y, p in zip(ys, ps) if y == "no" and p == "no")
    n = len(ys)
    return {
        "accuracy": round((tp + tn) / n, 4) if n else 0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (tp + fp + fn) else 0,
        "yes_ratio": round(sum(1 for p in ps if p == "yes") / n, 4),
        "total": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def run_qwen_presence(df, cache_path):
    print(f"  [Qwen] {len(df)} samples...")
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()
    qwen_proc = AutoProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
    yid = qwen_proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    nid = qwen_proc.tokenizer.encode("No", add_special_tokens=False)[0]

    cache = []
    t0 = time.time()
    for pos, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="  Qwen")):
        entry = {"pos": pos, "present": False, "logit_diff": -999, "n_boxes": 0}
        try:
            image = pil_from_parquet(row.get("image"))
            if image is None:
                cache.append(entry); continue
            obj = extract_object(row["question"])

            prompt = f"Is there a {obj} in this image? Answer only Yes or No."
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}]}]
            txt = qwen_proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            ins = qwen_proc(text=[txt], images=[image], return_tensors="pt")
            ins = {k: v.to(qwen.device) for k, v in ins.items()}
            with torch.no_grad():
                out = qwen(**ins)
            diff = out.logits[0, -1, yid].item() - out.logits[0, -1, nid].item()

            g_prompt = (f"Detect {obj} in this image. Output JSON list with bbox_2d. "
                        f"If not visible, output [].")
            msgs2 = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": g_prompt}]}]
            txt2 = qwen_proc.apply_chat_template(
                msgs2, tokenize=False, add_generation_prompt=True)
            ins2 = qwen_proc(text=[txt2], images=[image], return_tensors="pt")
            ins2 = {k: v.to(qwen.device) for k, v in ins2.items()}
            with torch.no_grad():
                gen = qwen.generate(**ins2, max_new_tokens=256, do_sample=False)
            out_ids = [o[len(i):] for i, o in zip(ins2["input_ids"], gen)]
            raw = qwen_proc.batch_decode(
                out_ids, skip_special_tokens=False)[0]
            del ins, ins2, gen, out_ids

            jb = re.findall(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
            if not jb:
                jb = re.findall(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
            n_boxes = 0
            for blk in jb:
                try:
                    for it in json.loads(blk):
                        if "bbox_2d" in it:
                            n_boxes += 1
                except json.JSONDecodeError:
                    continue

            entry = {"pos": pos, "present": diff > 0,
                     "logit_diff": diff, "n_boxes": n_boxes}
        except Exception as e:
            entry["error"] = str(e)
        cache.append(entry)

    elapsed = time.time() - t0
    del qwen
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Qwen done: {elapsed:.0f}s "
          f"(present={sum(1 for x in cache if x.get('present'))})")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    return cache


def v9_fuse(llava_yes_logit, llava_no_logit, qw):
    llava_diff = llava_yes_logit - llava_no_logit
    qw_diff = qw.get("logit_diff", 0)
    qw_present = qw.get("present", False)
    qw_boxes = qw.get("n_boxes", 0)

    qw_evidence = math.tanh(qw_diff * ALPHA) * BETA_MAX

    if qw_present and qw_boxes > 0:
        qw_evidence += min(qw_boxes * DELTA_BOX, 0.6)

    if qw_diff < 0:
        qw_evidence *= GAMMA_NEG

    fused = llava_diff + qw_evidence
    return "yes" if fused > 0 else "no"


def run_llava_fusion(df, qwen_cache):
    print(f"  [LLaVA] {len(df)} samples (baseline + v9)...")
    llava = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_PATH, torch_dtype=torch.float16, device_map="auto").eval()
    llava_proc = LlavaProcessor.from_pretrained(LLAVA_PATH)
    yid = llava_proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    nid = llava_proc.tokenizer.encode("No", add_special_tokens=False)[0]

    base_preds, gs_preds = [], []
    t0 = time.time()
    for pos, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="  LLaVA")):
        try:
            image = pil_from_parquet(row.get("image"))
            if image is None:
                base_preds.append("yes"); gs_preds.append("yes"); continue
            q = row["question"]
            prompt = (
                "<image>\nAnswer the question with a single word: Yes or No.\n"
                f"Question: {q}\nAnswer:")
            ins = llava_proc(text=prompt, images=image, return_tensors="pt")
            ins = {k: v.to(llava.device) for k, v in ins.items()}
            with torch.no_grad():
                out = llava(**ins)
            cy = out.logits[0, -1, yid].item()
            cn = out.logits[0, -1, nid].item()
            base_preds.append("yes" if cy > cn else "no")

            qw = qwen_cache[pos] if pos < len(qwen_cache) else {}
            gs_preds.append(v9_fuse(cy, cn, qw))
        except Exception:
            base_preds.append("yes")
            gs_preds.append("yes")

    elapsed = time.time() - t0
    del llava
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  LLaVA done: {elapsed:.0f}s ({len(df)/elapsed:.1f} it/s)")
    return metrics(df, base_preds), metrics(df, gs_preds)


def main():
    parser = argparse.ArgumentParser(description="V9 GS: Logit-Space Fusion")
    parser.add_argument("--splits", type=str, default="random,popular,adversarial",
                        help="Comma-separated splits")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit samples per split (None=all)")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    os.makedirs(V9_OUTPUT, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",")]
    all_results = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"POPE/{split}  ({args.n or 'all'} samples)")
        print(f"{'='*60}")

        df = load_pope(split, args.n)

        cache_p = os.path.join(V9_OUTPUT, f"qwencache_{split}.json")
        if os.path.exists(cache_p):
            with open(cache_p) as f:
                qc = json.load(f)
            print(f"  Qwen cache loaded: {cache_p}")
        else:
            qc = run_qwen_presence(df, cache_p)

        base_m, gs_m = run_llava_fusion(df, qc)
        all_results[split] = {"baseline": base_m, "v9_gs": gs_m}

        print(f"\n  {'Metric':>12s}  {'Baseline':>12s}  "
              f"{'V9 GS':>12s}  {'Δ':>8s}")
        print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
        for k in ["accuracy", "precision", "recall", "f1", "yes_ratio"]:
            d = gs_m[k] - base_m[k]
            s = "+" if d >= 0 else ""
            print(f"  {k:>12s}  {base_m[k]:12.4f}  "
                  f"{gs_m[k]:12.4f}  {s}{d:+.4f}")
        for k in ["tp", "fp", "fn", "tn"]:
            d = gs_m[k] - base_m[k]
            s = "+" if d >= 0 else ""
            print(f"  {k:>12s}  {base_m[k]:12}  "
                  f"{gs_m[k]:12}  {s}{d:+d}")

    out_p = os.path.join(V9_OUTPUT, "results_v9.json")
    with open(out_p, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_p}")


if __name__ == "__main__":
    main()
```

- baseline 总体准确率 **85.8%**，总体 F1 **84.6%
- v9_gs 总体准确率 **87.7%**，总体 F1 **86.6%**
- v7_gs 总体准确率**87.69%**，总体 F1 **86.57%**
- ms_max 总体准确率**86.43%**，总体F1 **86.12%**
    "overall": {
        "tp": 4043,
        "tn": 3529,
        "fp": 971,
        "fn": 457,
        "count": 9000,
        "accuracy": 0.8413333333333334,
        "precision": 0.8063422417231751,
        "recall": 0.8984444444444445,
        "f1": 0.8499054025646416
    }


#### POPE
##### BaseLine
```json
{
  "overall": {
    "accuracy": 0.8643,
    "precision": 0.8952,
    "recall": 0.8253,
    "f1": 0.8588,
    "yes_ratio": 0.461,
    "total": 9000,
    "tp": 3714,
    "fp": 435,
    "fn": 786,
    "tn": 4065
  },
  "by_split": {
    "adversarial": {
      "accuracy": 0.8343,
      "precision": 0.8405,
      "recall": 0.8253,
      "f1": 0.8328,
      "yes_ratio": 0.491,
      "total": 3000,
      "tp": 1238,
      "fp": 235,
      "fn": 262,
      "tn": 1265
    },
    "popular": {
      "accuracy": 0.868,
      "precision": 0.9023,
      "recall": 0.8253,
      "f1": 0.8621,
      "yes_ratio": 0.4573,
      "total": 3000,
      "tp": 1238,
      "fp": 134,
      "fn": 262,
      "tn": 1366
    },
    "random": {
      "accuracy": 0.8907,
      "precision": 0.9494,
      "recall": 0.8253,
      "f1": 0.883,
      "yes_ratio": 0.4347,
      "total": 3000,
      "tp": 1238,
      "fp": 66,
      "fn": 262,
      "tn": 1434
    }
  },
  "n_samples": 9000
}
```
##### LoRA
```json
{
  "overall": {
    "accuracy": 0.8758,
    "precision": 0.9251,
    "recall": 0.8178,
    "f1": 0.8681,
    "yes_ratio": 0.442,
    "total": 9000,
    "tp": 3680,
    "fp": 298,
    "fn": 820,
    "tn": 4202
  },
  "by_split": {
    "adversarial": {
      "accuracy": 0.8537,
      "precision": 0.8803,
      "recall": 0.8187,
      "f1": 0.8484,
      "yes_ratio": 0.465,
      "total": 3000,
      "tp": 1228,
      "fp": 167,
      "fn": 272,
      "tn": 1333
    },
    "popular": {
      "accuracy": 0.8747,
      "precision": 0.9232,
      "recall": 0.8173,
      "f1": 0.867,
      "yes_ratio": 0.4427,
      "total": 3000,
      "tp": 1226,
      "fp": 102,
      "fn": 274,
      "tn": 1398
    },
    "random": {
      "accuracy": 0.899,
      "precision": 0.9769,
      "recall": 0.8173,
      "f1": 0.89,
      "yes_ratio": 0.4183,
      "total": 3000,
      "tp": 1226,
      "fp": 29,
      "fn": 274,
      "tn": 1471
    }
  },
  "n_samples": 9000
}
```
#### MME
##### BaseLine
```json
{
  "overall": {
    "accuracy": 0.7905,
    "precision": 0.7768,
    "recall": 0.8151,
    "f1": 0.7955,
    "yes_ratio": 0.5247,
    "total": 2434,
    "tp": 992,
    "fp": 285,
    "fn": 225,
    "tn": 932
  },
  "by_category": {
    "existence": {
      "accuracy": 0.9667,
      "precision": 0.9667,
      "recall": 0.9667,
      "f1": 0.9667,
      "yes_ratio": 0.5,
      "total": 120,
      "tp": 58,
      "fp": 2,
      "fn": 2,
      "tn": 58
    },
    "code_reasoning": {
      "accuracy": 0.525,
      "precision": 0.5294,
      "recall": 0.45,
      "f1": 0.4865,
      "yes_ratio": 0.425,
      "total": 40,
      "tp": 9,
      "fp": 8,
      "fn": 11,
      "tn": 12
    },
    "artwork": {
      "accuracy": 0.6975,
      "precision": 0.6667,
      "recall": 0.79,
      "f1": 0.7231,
      "yes_ratio": 0.5925,
      "total": 400,
      "tp": 158,
      "fp": 79,
      "fn": 42,
      "tn": 121
    },
    "celebrity": {
      "accuracy": 0.7559,
      "precision": 0.7122,
      "recall": 0.8588,
      "f1": 0.7787,
      "yes_ratio": 0.6029,
      "total": 340,
      "tp": 146,
      "fp": 59,
      "fn": 24,
      "tn": 111
    },
    "numerical_calculation": {
      "accuracy": 0.425,
      "precision": 0.2857,
      "recall": 0.1,
      "f1": 0.1481,
      "yes_ratio": 0.175,
      "total": 40,
      "tp": 2,
      "fp": 5,
      "fn": 18,
      "tn": 15
    },
    "text_translation": {
      "accuracy": 0.525,
      "precision": 1.0,
      "recall": 0.05,
      "f1": 0.0952,
      "yes_ratio": 0.025,
      "total": 40,
      "tp": 1,
      "fp": 0,
      "fn": 19,
      "tn": 20
    },
    "count": {
      "accuracy": 0.8167,
      "precision": 0.8065,
      "recall": 0.8333,
      "f1": 0.8197,
      "yes_ratio": 0.5167,
      "total": 60,
      "tp": 25,
      "fp": 6,
      "fn": 5,
      "tn": 24
    },
    "color": {
      "accuracy": 0.8333,
      "precision": 0.7632,
      "recall": 0.9667,
      "f1": 0.8529,
      "yes_ratio": 0.6333,
      "total": 60,
      "tp": 29,
      "fp": 9,
      "fn": 1,
      "tn": 21
    },
    "commonsense_reasoning": {
      "accuracy": 0.7143,
      "precision": 0.6974,
      "recall": 0.7571,
      "f1": 0.726,
      "yes_ratio": 0.5429,
      "total": 140,
      "tp": 53,
      "fp": 23,
      "fn": 17,
      "tn": 47
    },
    "position": {
      "accuracy": 0.6833,
      "precision": 0.6279,
      "recall": 0.9,
      "f1": 0.7397,
      "yes_ratio": 0.7167,
      "total": 60,
      "tp": 27,
      "fp": 16,
      "fn": 3,
      "tn": 14
    },
    "OCR": {
      "accuracy": 0.825,
      "precision": 0.8095,
      "recall": 0.85,
      "f1": 0.8293,
      "yes_ratio": 0.525,
      "total": 40,
      "tp": 17,
      "fp": 4,
      "fn": 3,
      "tn": 16
    },
    "landmark": {
      "accuracy": 0.8625,
      "precision": 0.8008,
      "recall": 0.965,
      "f1": 0.8753,
      "yes_ratio": 0.6025,
      "total": 400,
      "tp": 193,
      "fp": 48,
      "fn": 7,
      "tn": 152
    },
    "scene": {
      "accuracy": 0.8775,
      "precision": 0.8953,
      "recall": 0.855,
      "f1": 0.8747,
      "yes_ratio": 0.4775,
      "total": 400,
      "tp": 171,
      "fp": 20,
      "fn": 29,
      "tn": 180
    },
    "posters": {
      "accuracy": 0.8299,
      "precision": 0.945,
      "recall": 0.7007,
      "f1": 0.8047,
      "yes_ratio": 0.3707,
      "total": 294,
      "tp": 103,
      "fp": 6,
      "fn": 44,
      "tn": 141
    }
  },
  "n_samples": 2434
}
```
##### LoRA
```json
{
  "overall": {
    "accuracy": 0.7806,
    "precision": 0.7916,
    "recall": 0.7588,
    "f1": 0.7749,
    "yes_ratio": 0.4769,
    "total": 1600,
    "tp": 604,
    "fp": 159,
    "fn": 192,
    "tn": 645
  },
  "by_category": {
    "existence": {
      "accuracy": 0.95,
      "precision": 1.0,
      "recall": 0.9,
      "f1": 0.9474,
      "yes_ratio": 0.45,
      "total": 120,
      "tp": 54,
      "fp": 0,
      "fn": 6,
      "tn": 60
    },
    "code_reasoning": {
      "accuracy": 0.5,
      "precision": 0.5, 
      "recall": 0.5,
      "f1": 0.5,
      "yes_ratio": 0.5,
      "total": 40,
      "tp": 10,
      "fp": 10,
      "fn": 10,
      "tn": 10
    },
    "artwork": {
      "accuracy": 0.74,
      "precision": 0.7143,
      "recall": 0.7732,
      "f1": 0.7426,
      "yes_ratio": 0.525,
      "total": 200,
      "tp": 75,
      "fp": 30,
      "fn": 22,
      "tn": 73
    },
    "celebrity": {
      "accuracy": 0.745,
      "precision": 0.7069,
      "recall": 0.8283,
      "f1": 0.7628,
      "yes_ratio": 0.58,
      "total": 200,
      "tp": 82,
      "fp": 34,
      "fn": 17,
      "tn": 67
    },
    "numerical_calculation": {
      "accuracy": 0.325,
      "precision": 0.1111,
      "recall": 0.05,
      "f1": 0.069,
      "yes_ratio": 0.225,
      "total": 40,
      "tp": 1,
      "fp": 8,
      "fn": 19,
      "tn": 12
    },
    "text_translation": {
      "accuracy": 0.5,
      "precision": 0,
      "recall": 0.0,
      "f1": 0.0,
      "yes_ratio": 0.0,
      "total": 40,
      "tp": 0,
      "fp": 0,
      "fn": 20,
      "tn": 20
    },
    "count": {
      "accuracy": 0.7833,
      "precision": 0.7429,
      "recall": 0.8667,
      "f1": 0.8,
      "yes_ratio": 0.5833,
      "total": 60,
      "tp": 26,
      "fp": 9,
      "fn": 4,
      "tn": 21
    },
    "color": {
      "accuracy": 0.85,
      "precision": 0.7692,
      "recall": 1.0,
      "f1": 0.8696,
      "yes_ratio": 0.65,
      "total": 60,
      "tp": 30,
      "fp": 9,
      "fn": 0,
      "tn": 21
    },
    "commonsense_reasoning": {
      "accuracy": 0.7429,
      "precision": 0.75,
      "recall": 0.7286,
      "f1": 0.7391,
      "yes_ratio": 0.4857,
      "total": 140,
      "tp": 51,
      "fp": 17,
      "fn": 19,
      "tn": 53
    },
    "position": {
      "accuracy": 0.7333,
      "precision": 0.675,
      "recall": 0.9,
      "f1": 0.7714,
      "yes_ratio": 0.6667,
      "total": 60,
      "tp": 27,
      "fp": 13,
      "fn": 3,
      "tn": 17
    },
    "OCR": {
      "accuracy": 0.675,
      "precision": 0.6207,
      "recall": 0.9,
      "f1": 0.7347,
      "yes_ratio": 0.725,
      "total": 40,
      "tp": 18,
      "fp": 11,
      "fn": 2,
      "tn": 9
    },
    "landmark": {
      "accuracy": 0.905,
      "precision": 0.9062,
      "recall": 0.8969,
      "f1": 0.9016,
      "yes_ratio": 0.48,
      "total": 200,
      "tp": 87,
      "fp": 9,
      "fn": 10,
      "tn": 94
    },
    "scene": {
      "accuracy": 0.84,
      "precision": 0.9221,
      "recall": 0.732,
      "f1": 0.8161,
      "yes_ratio": 0.385,
      "total": 200,
      "tp": 71,
      "fp": 6,
      "fn": 26,
      "tn": 97
    },
    "posters": {
      "accuracy": 0.815,
      "precision": 0.96,
      "recall": 0.6792,
      "f1": 0.7956,
      "yes_ratio": 0.375,
      "total": 200,
      "tp": 72,
      "fp": 3,
      "fn": 34,
      "tn": 91
    }
  },
  "n_samples": 1600
}
```

#### AMBER
##### BaseLine
```json
{
  "generative": {
    "metrics": {
      "note": "full CHAIR requires COCO captions; this is a simplified version"
    },
    "n_samples": 1004
  },
  "discriminative": {
    "metrics": {
      "accuracy": 0.8062,
      "precision": 0.7073,
      "recall": 0.7137,
      "f1": 0.7105,
      "yes_ratio": 0.3362,
      "total": 5000,
      "tp": 1189,
      "fp": 492,
      "fn": 477,
      "tn": 2842
    },
    "n_samples": 5000
  }
}
```
##### LoRA
```json
{
  "generative": {
    "metrics": {
      "note": "full CHAIR requires COCO captions; this is a simplified version"
    },
    "n_samples": 1004
  },
  "discriminative": {
    "metrics": {
      "accuracy": 0.8292,
      "precision": 0.7074,
      "recall": 0.8313,
      "f1": 0.7643,
      "yes_ratio": 0.3916,
      "total": 5000,
      "tp": 1385,
      "fp": 573,
      "fn": 281,
      "tn": 2761
    },
    "n_samples": 5000
  }
}
```







根据OPA-DPO，我们知道模型


根据实验我发现：抑制背景能显著提升判别式任务，抑制前景能显著提升生成式任务。

attention_flow_heatmap.png
通过实验分析：
- LM15层能定位到物体（相近的物体也会发现），这里决定了后续是否能正确识别物体;
- LM19层能判断其他语义，如颜色，位置;
- LM31层一定会再反方向语义上拉扯一下。
应该怎么做：
对15层：主要是锚定物体的，如果能锚定到物体，那么再判断LM19层的语义（空间，颜色，数量），但是锚定的物体可能会出错，如spoon这个物体太小了，锚定出错了！否则认为该物体不存在。
对19层：他会结合语义信息初步判断，但是如果15层锚定的物体与语义物体相近，很可能会判断出错。


场景中如果出现了相似物体或常见物体周围的东西 --> 诱导模型先验知识 压过 判断力。
场景中如果没有相似物体

vision_cls_per_layer.png
在视觉层，感觉1,2,4,8层就能把前背景分离出来，观察这个图片是否可以将这几个层的作为视觉增强让模型能更强的检测出图片中出现的各种物体。

我发现了一个问题，根据 `/home/jay/Desktop/code/Project/outputs/cross_modal_match/single/` `/home/jay/Desktop/code/Project/outputs/single/` 反映出的问题：
就是其实就是对于右上角的勺子太小了，图片整体信息盖过了注意力，如果给特定区域裁剪输入，其应该是能确认物体信息的，我目前的思路是对其进行标准的九宫格裁剪，然后判断某个或这某几个格子是否注意到了问题当中提到的物体，如果提到了就增强该信息，如果全部格子都没注意到那么就一定没有该物体或者相似的物体，我觉得本质上是对照片的各个位置进行放大观察，你觉得这个思路怎么样，该怎么做，或者还有其他优化方法？



##### 结论
L15量化「物体匹配度」;
L19根据问题类型做不同深度的语义推理;

    [L16+L17] → "Yes, there is a spoon in the image, placed on the plate with the sandwich and salad."
    [L16+L17+L18] → "Yes, there is a spoon in the image, which is placed on the plate with the sandwich and salad."
    [L16+L17+L18+L19] → "Yes, there is a spoon in the image. It is placed on the plate with the sandwich and salad."
    [L15+L16+L17] → "Yes, there is a spoon in the image."
    [L15+L16+L17+L18+L19] → "Yes, a spoon is present in the image. It is placed next to the sandwich on the plate."


我想
```python
通过实验分析：
- 图片中存在的物体和问题的物体具有高相似度或一致：
	- 图片在LM 15左右一定会定位到该物体附近;
	否则其整体的注意力全局就比较一样，决策就偏向绝对。
	这一步的决策对后续是否有幻觉有关键性作用。
```

```PlainText
之前 (V1-V3):
  稳定性投票 → vote_counts → 直接用投票数做 gate weights
                 ↑ k=1, 每个 channel 选 1 个 patch

V4:
  稳定性投票(k=7) → 池化选中的 patches → stability-pooled CLS proxy
                  → cosine_sim(proxy, 每个patch) → gate weights
                  ↑ 核心：用 LAST-ViT 的 "更好 CLS" 指导 gating
```
#### 数据集分布
##### LLaVA-Instruct-150K
各个数据集分布：

|数据集名称|样本数|唯一图像数|平均对话轮次|平均Human文本长度|平均GPT文本长度|
|---|---|---|---|---|---|
|llava_instruct_150k|157,712|81,479|2.29|57.3|407.1|
|llava_instruct_80k|80,000|80,000|2.77|57.3|372.1|
|conversation_58k|56,681|56,681|4.53|53.4|291.1|
|detail_23k|23,240|23,240|1.00|44.5|609.3|
|complex_reasoning_77k|76,643|76,643|1.00|74.6|736.0|
|llava_v1_5_mix665k|665,298|349,034|5.18|67.7|80.8|

  数据集特点：
  - detail_23k 和 complex_reasoning_77k：单轮对话，GPT回答较长（600-700字符），适合详细描述和复杂推理任务
  - conversation_58k：多轮对话（平均4.5轮），适合对话式交互训练
  - llava_v1_5_mix665k：最大数据集，包含大量简短回答（中位数仅14字符）
###### POPE
各数据集：

| 数据集/分割             | 样本数   | 唯一图像数 | Yes比例 | No比例  | 唯一物体数 |
| ------------------ | ----- | ----- | ----- | ----- | ----- |
| test (default)     | 9,000 | 500   | 50.0% | 50.0% | 79    |
| Full (all splits)  | 9,000 | 500   | 50.0% | 50.0% | 79    |
| Full - adversarial | 3,000 | 500   | 50.0% | 50.0% | 74    |
| Full - popular     | 3,000 | 500   | 50.0% | 50.0% | 79    |
| Full - random      | 3,000 | 500   | 50.0% | 50.0% | 79    |
  数据集特点：
  - 任务类型：物体幻觉评估
  - 问题格式："Is there a [object] in the image?"
  - 答案类型：二分类                             
  - 数据来源：COCO数据集  
  三个分割的区别：
  - adversarial：负样本来自图像中存在的物体，更具挑战性
  - popular：负样本来自高频物体
  - random：负样本随机选择
##### MME
整体统计：
![](assets/Idea/file-20260508112638554.png#pig_center)
类别分布：
![](assets/Idea/file-20260508112706349.png#pig_center)
  数据集特点：    
  - 任务类型：多模态评估基准
  - 评估维度：感知（perception）和认知（cognition）
  - 问题类型：全部为 Yes/No 二分类问题
  - 答案分布：完全平衡，Yes/No 各占50%  

##### AMBER
各数据集：
![](assets/Idea/file-20260508113402479.png#pig_center)
任务分布：

| 任务类型     | 样本数    | 描述       |
| -------- | ------ | -------- |
| 生成任务     | 1,004  | 图像描述生成   |
| 判别任务-存在性 | ~5,000 | 判断物体是否存在 |
| 判别任务-属性  | ~8,000 | 判断物体属性   |
| 判别任务-关系  | ~1,200 | 判断物体关系   |
  数据集特点：
  - 任务类型：生成任务 + 判别任务
  - 评估维度：存在性、属性、关系三个维度
  - 幻觉检测：标注了真实物体和幻觉物体  


#### 测试
在幻觉检测数据集中测试文本token 和 视觉token 对模型推理时的贡献：



#### Qwen2.5-VL 3B
Sample 3000 
###### POPE
```python
===== POPE Results (peft) =====
        random | Acc: 0.8990 | F1: 0.8900 | P: 0.9769 | R: 0.8173
       popular | Acc: 0.8847 | F1: 0.8763 | P: 0.9445 | R: 0.8173
   adversarial | Acc: 0.8670 | F1: 0.8600 | P: 0.9075 | R: 0.8173
       overall | Acc: 0.8836 | F1: 0.8753 | P: 0.9421 | R: 0.8173    
==================================================
POPE Evaluation (Residual LAST-ViT v2)
==================================================
Total POPE samples: 9000
POPE v2: 100%|██████████████████████████████| 9000/9000 [32:37<00:00,  4.60it/s]
        random | Acc: 0.8947 | F1: 0.8839 | P: 0.9845 | R: 0.8020
       popular | Acc: 0.8823 | F1: 0.8721 | P: 0.9555 | R: 0.8020
   adversarial | Acc: 0.8620 | F1: 0.8531 | P: 0.9120 | R: 0.8013
       overall | Acc: 0.8797 | F1: 0.8695 | P: 0.9497 | R: 0.8018
       
       
======================================================================
POPE Three-Way Comparison
======================================================================
    Category |   Baseline |  LAST-ViT v1 |  Residual v2 |     Δ v1 |     Δ v2
----------------------------------------------------------------------
	     overall |     0.8798 |       0.8836 |       0.8797 |  +0.0038 |  -0.0001
      random |     0.8913 |       0.8990 |       0.8947 |  +0.0077 |  +0.0034
     popular |     0.8813 |       0.8847 |       0.8823 |  +0.0034 |  +0.0010
 adversarial |     0.8667 |       0.8670 |       0.8620 |  +0.0003 |  -0.0047
```
##### MME
```python
===== MME Results (baseline) =====
  Overall Accuracy: 0.8538 (2027/2374)
                   OCR: 0.7250 (29/40)
               artwork: 0.8150 (326/400)
             celebrity: 0.7912 (269/340)
        code_reasoning: 0.8750 (35/40)
                 color: 0.9500 (57/60)
  commonsense_reasoning: 0.7429 (104/140)
                 count: 0.8667 (52/60)
             existence: 0.9500 (57/60)
              landmark: 0.9325 (373/400)
  numerical_calculation: 0.7750 (31/40)
              position: 0.8500 (51/60)
               posters: 0.9082 (267/294)
                 scene: 0.8400 (336/400)
      text_translation: 1.0000 (40/40)
===== MME Results (peft) =====
  Overall Accuracy: 0.8433 (2002/2374)
                   OCR: 0.6750 (27/40)
               artwork: 0.7375 (295/400)
             celebrity: 0.7588 (258/340)
        code_reasoning: 0.8750 (35/40)
                 color: 0.9000 (54/60)
  commonsense_reasoning: 0.7786 (109/140)
                 count: 0.8833 (53/60)
             existence: 0.9667 (58/60)
              landmark: 0.9450 (378/400)
  numerical_calculation: 0.8000 (32/40)
              position: 0.8333 (50/60)
               posters: 0.9048 (266/294)
                 scene: 0.8675 (347/400)
      text_translation: 1.0000 (40/40)
      
==================================================
MME Evaluation (Residual LAST-ViT v2)
==================================================
 Overall: 0.8614 (2045/2374)
                   OCR: 0.7500 (30/40)
               artwork: 0.8000 (320/400)
             celebrity: 0.8353 (284/340)
        code_reasoning: 0.8500 (34/40)
                 color: 0.9500 (57/60)
  commonsense_reasoning: 0.7857 (110/140)
                 count: 0.8833 (53/60)
             existence: 0.9500 (57/60)
              landmark: 0.9350 (374/400)
  numerical_calculation: 0.7250 (29/40)
              position: 0.8500 (51/60)
               posters: 0.9082 (267/294)
                 scene: 0.8475 (339/400)
      text_translation: 1.0000 (40/40)
======================================================================
MME Three-Way Comparison
======================================================================
      Metric |   Baseline |  LAST-ViT v1 |  Residual v2 |     Δ v1 |     Δ v2
----------------------------------------------------------------------
     overall |     0.8538 |       0.8433 |       0.8614 |  -0.0105 |  +0.0076
  perception |     0.8595 |       0.8448 |       0.8666 |  -0.0147 |  +0.0071
   cognition |     0.8077 |       0.8308 |       0.8192 |  +0.0231 |  +0.0115
```
##### AMBER
```python
--- Baseline ---
  Generative: CHAIR=0.0070 Cover=0.6116 Hal=0.0372 Cog=0.7872
  Discriminative: Overall=0.8760 Existence=0.9580 Attr=0.8349 Relation=0.9647
  
--- LAST-ViT v1 ---
  Generative: CHAIR=0.0099 Cover=0.4219 Hal=0.0134 Cog=0.7042
  Discriminative: Overall=0.8470 Existence=0.9147 Attr=0.8064 Relation=0.8824
  
--- Residual v2 ---
  Generative: CHAIR=0.0111 Cover=0.4232 Hal=0.0139 Cog=0.7047
  Discriminative: Overall=0.8645 Existence=0.9441 Attr=0.8294 Relation=0.9059
===========================================================================
AMBER Three-Way Comparison
===========================================================================
                      Metric |   Baseline |  LAST-ViT v1 |  Residual v2
---------------------------------------------------------------------------
Generative CHAIR (lower better) |     0.0070 |     0.0099 ★ |     0.0111
            Generative Cover |     0.6116 |     0.4219   |     0.4232 ★
Generative Hal (lower better) |     0.0372 |     0.0134 ★ |     0.0139
              Generative Cog |     0.7872 |     0.7042   |     0.7047 ★
       Discrim Existence Acc |     0.9580 |     0.9147   |     0.9441 ★
       Discrim Attribute Acc |     0.8349 |     0.8064   |     0.8294 ★
        Discrim Relation Acc |     0.9647 |     0.8824   |     0.9059 ★
         Discrim Overall Acc |     0.8760 |     0.8470   |     0.8645 ★
```