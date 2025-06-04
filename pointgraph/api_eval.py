import os
import re
import json
import argparse
from PIL import Image
from tqdm import tqdm

from depth import metric3dv2 as depth_model
from serve.scene_graph import get_scene_graph
from segmentation import sam, florence as detection

from utils import vqa_parsing, llm_judge, make_client
from serve.system_prompts import object_parsing_prompt, vqa_reasoning_prompt, FORMAT_PROMPTS


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
    parser.add_argument("--dataset_path", type=str, default="../dataset")
    parser.add_argument("--result_path", default="result")
    parser.add_argument("--model_id", default="gpt-4.1")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--eval_type", choices=["re", "json", "llm", "direct"], default="re")
    parser.add_argument("--group_index", type=int, default=0) # parallel eval index of sub-process
    parser.add_argument("--group", type=int, default=1) # parallel eval number of sub-process
    # parallel eval parameters, if not use parallel eval, set group_index to 0 and group to 1
    args = parser.parse_args()

    model_id = args.model_id
    eval_type = args.eval_type
    dataset_path = args.dataset_path
    result_path = os.path.join(args.result_path, model_id)
    os.makedirs(result_path, exist_ok=True)
    output_file = os.path.join(result_path, f"results_{args.group_index}.json")
    print('output_file: ', output_file)

    print("Load models...")
    client = make_client()

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    metriced_model = depth_model.get_model()

    result = blank_stats()

    info_list = json.load(open(os.path.join(dataset_path, 'data.json'))) * args.repeat
    info_list = info_list[args.group_index::args.group]
    total = len(info_list)
    print("Total: ", total)

    for info in tqdm(info_list):
        raw_id = info["id"]
        question = info["question"]
        options = info["options"]
        gt_idx = info["answer"]
        task_type = info["task_type"]
        sub_task = info["sub_task_type"]

        img_path = os.path.join(dataset_path, task_type, f"{raw_id.split('_')[0]}.png")
        image = Image.open(img_path).convert("RGB")

        prompt = "Question: " + question
        for i in range(len(options)):
            prompt += f"\n{chr(65 + i)}. {options[i]}"

        object_list = vqa_parsing("Question: " + question, image, sys_prompt=object_parsing_prompt, model=model_id, client=client)
        detections = detection.get_detections(image, object_list, detection_model, result_path)
        mask, _, object_names = sam.get_mask(image, object_list, sam_model, detections, result_path)

        _, _, pcd = depth_model.depth_estimation(image, metriced_model, result_path)
        scene_graph, _ = get_scene_graph(image, pcd, mask, object_names)

        prompt += f"Scene graph: {json.dumps(scene_graph, indent=2)}\n"
        response = vqa_parsing(prompt, image, sys_prompt=vqa_reasoning_prompt + '\n' + FORMAT_PROMPTS[eval_type], model=model_id, client=client)
        print(response)

        # ---------------------- evaluation -------------------------------
        gt_letter = chr(65 + gt_idx).upper()
        if eval_type == "json":
            try:
                cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()
                pred_letter = json.loads(cleaned).get("answer", "A").strip().upper()[:1]
            except Exception:
                pred_letter = "A"
            flag = pred_letter == gt_letter
        elif eval_type == "re":
            PATTERN = re.compile(r"Answer\s*:\s*([A-D])\b", re.IGNORECASE)
            pred_letter = PATTERN.findall(response)[-1] if len(PATTERN.findall(response)) > 0 else "A"
            flag = pred_letter == gt_letter
        elif eval_type == "direct":
            pred_letter = response.strip().upper()[:1]
            flag = pred_letter == gt_letter
        elif eval_type == "llm":
            flag = llm_judge(question=prompt, pred=response, gt=gt_letter, client=client, judge_model="gpt-4.1-mini")
        else:
            assert False, f"Unknown eval_type: {eval_type}"

        result["Total"].append(flag)
        result[task_type][sub_task].append(flag)
        result[task_type]["Total"].append(flag)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

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
