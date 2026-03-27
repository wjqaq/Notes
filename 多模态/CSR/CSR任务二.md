# 关键代码


## CSR的数据集生成(用vllm部署模型 + openai 库调用)
```python
import os
import re
import json
import argparse
import base64
from io import BytesIO
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import glob


def encode_image_to_base64(image, format='PNG', max_len=1344, min_len=672):
    def expand2square(pil_img, background_color=(122, 116, 104)):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    if max(image.size) > max_len:
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))

    buffered = BytesIO()
    image.save(buffered, format=format)
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str


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


class Conversation:
    def __init__(self, system="", roles=("USER", "ASSISTANT"), version="v1"):
        self.system = system
        self.roles = roles
        self.messages = []
        self.offset = 0
        self.version = version

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            version=self.version
        )

    def get_prompt(self):
        ret = self.system
        for role, message in self.messages:
            if message:
                ret += role + ": " + message + " "
            else:
                ret += role + ": "
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])


conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
)

conv_templates = {
    "v1": conv_vicuna_v1,
}


class Node:
    def __init__(self, text, score, depth, parent=None, is_final=False):
        self.text = text
        self.score = score
        self.depth = depth
        self.parent = parent
        self.children = []
        self.is_final = is_final

    def add_child(self, child):
        self.children.append(child)


def call_vllm_api(prompt, image_b64, api_url="http://localhost:8000/v1/chat/completions",
                  model_name="llava-1.5-7b-hf", max_tokens=70, temperature=0.7,
                  num_beams=5, eos_token_id=29889):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "num_beams": num_beams,
        "eos_token_id": eos_token_id,
    }

    response = requests.post(api_url, headers=headers, json=data, timeout=120)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")


def sentence_level_beam_search_tree(qid, initial_text, image_b64, api_url, model_name,
                                     sentence_end_id, max_length, max_new_tokens,
                                     num_beams, num_beam_group, token_level_beams,
                                     diversity_penalty, max_depth=10):
    root = Node(initial_text, 0, 0)
    active_nodes = [root]
    iteration = 0

    with ThreadPoolExecutor(max_workers=token_level_beams) as executor:
        while active_nodes and iteration < max_depth:
            iteration += 1
            new_nodes = []
            futures = {}

            for node in active_nodes:
                if node.depth >= max_depth:
                    continue
                inputs = {
                    "prompt": node.text,
                    "image_b64": image_b64,
                    "max_new_tokens": max_new_tokens,
                    "sentence_end_id": sentence_end_id,
                    "num_beams": token_level_beams,
                    "diversity_penalty": diversity_penalty,
                }
                future = executor.submit(call_beam_search_step, inputs, api_url, model_name, node, image_b64)
                futures[future] = node

            for future in as_completed(futures):
                node = futures[future]
                try:
                    results = future.result(timeout=60)
                    if not results:
                        continue
                    for text, score, is_final in results:
                        new_score = node.score + score
                        new_node = Node(text, new_score, node.depth + 1, node, is_final)
                        node.add_child(new_node)
                        if not is_final and new_node.depth < max_depth:
                            new_nodes.append(new_node)
                except Exception as e:
                    print(f"Error in beam search step: {e}")
                    continue

            if not new_nodes:
                break

            new_nodes.sort(key=lambda x: x.score, reverse=True)
            active_nodes = new_nodes[:int(num_beams/2)-1] + new_nodes[-int(num_beams/2):] if len(new_nodes) >= num_beams else new_nodes

    return [{'id': qid, 'tree': root}]


def call_beam_search_step(inputs, api_url, model_name, parent_node, image_b64):
    prompt = inputs["prompt"]
    max_new_tokens = inputs["max_new_tokens"]
    num_beams = inputs["num_beams"]
    diversity_penalty = inputs["diversity_penalty"]

    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "n": num_beams,
        "diversity_penalty": diversity_penalty,
    }

    response = requests.post(api_url, headers=headers, json=data, timeout=120)
    if response.status_code != 200:
        return []

    result = response.json()
    contents = result['choices']

    results = []
    for i, choice in enumerate(contents):
        text = choice['message']['content']
        score = -i / num_beams
        is_final = len(text) >= 1024 or i == 0
        results.append((text, score, is_final))

    return results


def save_object(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_pickles(folder_path):
    pickle_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                pickle_list.append(data)
    return pickle_list


def eval_model(args):
    api_url = args.api_url
    model_name = args.model_name
    output_dir = args.output_dir
    dataset_path = args.dataset_path
    images_dir = args.images_dir

    with open(dataset_path, 'r', encoding='utf8') as fp:
        my_dataset = json.load(fp)

    for data_item in tqdm(my_dataset, desc="Processing"):
        qid = data_item['id']
        input_question = data_item['input']
        image_path = data_item['image']

        input_question_clean = input_question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
        prompts = get_prompts([input_question_clean])
        prompt = prompts[0]

        image_filename = 'COCO_train2014_' + image_path if not image_path.startswith('COCO_train2014_') else image_path
        full_image_path = os.path.join(images_dir, image_filename)
        image = Image.open(full_image_path)
        image_b64 = encode_image_to_base64(image)

        result = sentence_level_beam_search_tree(
            qid,
            prompt,
            image_b64,
            api_url,
            model_name,
            sentence_end_id=int(args.period_id),
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            num_beams=int(args.num_beams),
            num_beam_group=int(args.num_beam_group),
            token_level_beams=int(args.num_token_beams),
            diversity_penalty=float(args.diversity_penalty),
            max_depth=10
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for obj in result:
            save_path = os.path.join(output_dir, str(obj['id']) + '.pkl')
            save_object(obj, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="vLLM API server URL")
    parser.add_argument("--model_name", type=str, default="llava-1.5-7b-hf",
                        help="Model name served by vLLM")
    parser.add_argument("--dataset_path", type=str, default='./data/CSR-Prompt-Dataset-12k.json',
                        help="Path to the prompt dataset")
    parser.add_argument("--images_dir", type=str, default="./data/images",
                        help="Directory to images")
    parser.add_argument("--output_dir", type=str, default="./outputs/sample",
                        help="Path to save results")
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

#### openAI
## Qwen2-VL 的 Visual Contrastive Decoding(VCD)


# 成功运行截图