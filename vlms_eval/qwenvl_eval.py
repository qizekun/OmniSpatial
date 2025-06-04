import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from utils import SYS_PROMPTS, FORMAT_PROMPTS, make_client, llm_judge


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
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    # "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen2.5-VL-32B-Instruct" "Qwen/Qwen2.5-VL-72B-Instruct"
    # "remyxai/SpaceQwen2.5-VL-3B-Instruct" "remyxai/SpaceThinker-Qwen2.5VL-3B"
    parser.add_argument("--prompt_type", choices=["none", "zeroshot_cot", "manual_cot"], default="manual_cot")
    parser.add_argument("--eval_type", choices=["re", "json", "llm", "direct"], default="re")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--group_index", type=int, default=0) # parallel eval index of sub-process
    parser.add_argument("--group", type=int, default=1) # parallel eval number of sub-process
    # parallel eval parameters, if not use parallel eval, set group_index to 0 and group to 1
    parser.add_argument("--dataset_path", default="../dataset")
    parser.add_argument("--result_path", default="result")
    args = parser.parse_args()

    result_path = os.path.join(args.result_path, args.model_id)
    os.makedirs(result_path, exist_ok=True)
    output_file = os.path.join(result_path, f"results_{args.group_index}.json")
    print('output_file: ', output_file)

    model_id = args.model_id
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        load_in_8bit=False
    )
    processor = AutoProcessor.from_pretrained(model_id)

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
            prompt += f"\n{chr(65 + i)}. {options[i]}"

        image_path = os.path.join(args.dataset_path, task_type, f"{raw_id.split('_')[0]}.png")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
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
