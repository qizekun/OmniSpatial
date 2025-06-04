import os
import re
import math
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torchvision.transforms.functional import InterpolationMode
from utils import SYS_PROMPTS, FORMAT_PROMPTS, make_client, llm_judge

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def blank_stats():
    return {
        "Total": [],
        "Dynamic_Reasoning": {"Manipulation": [], "Motion_Analysis": [], "Total": []},
        "Spatial_Interaction": {"Traffic_Analysis": [], "Localization": [], "Geospatial_Strategy": [], "Total": []},
        "Complex_Logic": {"Pattern_Recognition": [], "Geometric_Reasoning": [], "Total": []},
        "Perspective_Taking": {"Egocentric": [], "Allocentric": [], "Hypothetical": [], "Total": []},
    }


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="OpenGVLab/InternVL3-8B")
    # 'OpenGVLab/InternVL3-2B' 'OpenGVLab/InternVL3-8B' 'OpenGVLab/InternVL3-9B'
    # 'OpenGVLab/InternVL3-14B' 'OpenGVLab/InternVL3-38B' 'OpenGVLab/InternVL3-78BB'
    parser.add_argument("--prompt_type", choices=["none", "zeroshot_cot", "manual_cot"], default="manual_cot")
    parser.add_argument("--eval_type", choices=["re", "json", "llm", "direct"], default="re")
    parser.add_argument("--group_index", type=int, default=0) # parallel eval index of sub-process
    parser.add_argument("--group", type=int, default=1) # parallel eval number of sub-process
    # parallel eval parameters, if not use parallel eval, set group_index to 0 and group to 1
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dataset_path", default="../dataset")
    parser.add_argument("--result_path", default="result")
    args = parser.parse_args()

    result_path = os.path.join(args.result_path, args.model_id)
    os.makedirs(result_path, exist_ok=True)
    output_file = os.path.join(result_path, f"results_{args.group_index}.json")
    print('output_file: ', output_file)


    # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
    # If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
    model_id = args.model_id
    device_map = split_model(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    data = json.load(open(os.path.join(args.dataset_path, 'data.json'))) * args.repeat
    data = data[args.group_index::args.group]
    total = len(data)
    print("Total: ", total)

    result = blank_stats()

    if args.eval_type == "llm":
        client = make_client()

    for info in tqdm(data):
        raw_id = info["id"]

        question = info["question"]
        options = info["options"]
        answer = info["answer"]
        task_type = info["task_type"]
        sub_task_type = info["sub_task_type"]

        prompt = SYS_PROMPTS[args.prompt_type] + '\n' + FORMAT_PROMPTS[args.eval_type] + '\n\n' + question
        for i in range(len(options)):
            prompt += f"{chr(65 + i)}. {options[i]}\n"

        image_path = os.path.join(args.dataset_path, task_type, f"{raw_id.split('_')[0]}.png")

        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=8192, do_sample=True)

        question = '<image>\n' + prompt
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        print(response)

        gt_letter = chr(65 + answer)
        if args.eval_type == "json":
            try:
                cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()
                pred_letter = json.loads(cleaned).get("answer", "A").strip().upper()[:1]
            except Exception:
                pred_letter = "A"
            flag = pred_letter == gt_letter
        elif args.eval_type == "re":
            PATTERN = re.compile(r"Answer\s*:\s*([A-D])\b", re.IGNORECASE)
            pred_letter = PATTERN.findall(response)[-1] if len(PATTERN.findall(response)) > 0 else "A"
            flag = pred_letter == gt_letter
        elif args.eval_type == "direct":
            pred_letter = response.strip().upper()[:1]
            flag = pred_letter == gt_letter
        elif args.eval_type == "llm":
            flag = llm_judge(question=prompt, pred=response, gt=gt_letter, client=client, judge_model="gpt-4.1-mini")
        else:
            assert False, f"Unknown eval_type: {args.eval_type}"
        
        result["Total"].append(flag)
        result[task_type][sub_task_type].append(flag)
        result[task_type]["Total"].append(flag)


    # final report -----------------------------------------------------------
    eps = 1e-6
    overall = sum(result["Total"]) / (len(result["Total"])+eps) * 100
    print("\n======= FINAL =======")
    print(f"Overall: {overall:.2f}% (N={len(result['Total'])})")
    for task in [k for k in result if k not in {"Total", "Processed"}]:
        task_acc = sum(result[task]["Total"]) / (len(result[task]["Total"])+eps) * 100
        print(f"{task}: {task_acc:.2f}%")
        for sub in result[task]:
            if sub == "Total":
                continue
            sub_acc = sum(result[task][sub]) / (len(result[task][sub])+eps) * 100
            print(f"    {sub}: {sub_acc:.2f}%")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
