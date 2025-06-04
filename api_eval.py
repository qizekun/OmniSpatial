import os
import re
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm
from openai import OpenAI

from utils import SYS_PROMPTS, FORMAT_PROMPTS, vqa_parsing, llm_judge, make_client, blind_eval


###############################################################################
#                          Single-sample processing                           #
###############################################################################

def process_info(info: dict, *, args: argparse.Namespace, client: OpenAI):
    response_id = info["id"]  # e.g. "15_1"
    question = info["question"]
    options = info["options"]
    gt_idx = info["answer"]
    task_type = info["task_type"]
    sub_task = info["sub_task_type"]
    repeat = info["repeat"]
    uid = f"{task_type}_{response_id}_{repeat}"

    model_id = args.model_id
    prompt_type = args.prompt_type
    eval_type = args.eval_type

    sys_prompt = SYS_PROMPTS[prompt_type] + '\n' + FORMAT_PROMPTS[eval_type]
    
    prompt = question + "\n"
    for i in range(len(options)):
        prompt += f"{chr(65 + i)}. {options[i]}\n"

    img_path = os.path.join(args.dataset_path, task_type, f"{response_id.split('_')[0]}.png")
    img = Image.open(img_path).convert("RGB")

    if args.blind:
        response = blind_eval(prompt, model_id, client=client)
    else:
        response = vqa_parsing(prompt, img, sys_prompt=sys_prompt, model=model_id, client=client)
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

    return flag, task_type, sub_task, uid


###############################################################################
#                     Aggregation helpers (+ resume logic)                    #
###############################################################################

def blank_stats():
    return {
        "Processed": [], # for resume
        "Total": [],
        "Dynamic_Reasoning": {"Manipulation": [], "Motion_Analysis": [], "Total": []},
        "Spatial_Interaction": {"Traffic_Analysis": [], "Localization": [], "Geospatial_Strategy": [], "Total": []},
        "Complex_Logic": {"Pattern_Recognition": [], "Geometric_Reasoning": [], "Total": []},
        "Perspective_Taking": {"Egocentric": [], "Allocentric": [], "Hypothetical": [], "Total": []},
    }


def safe_update_and_dump(res: dict, flag: bool, task: str, sub: str, uid: str, path: str, lock: threading.Lock):
    with lock:
        res["Processed"].append(uid)
        res["Total"].append(flag)
        res[task][sub].append(flag)
        res[task]["Total"].append(flag)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(res, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch VQA evaluation helper")
    parser.add_argument("--model_id", default="gpt-4.1")
    parser.add_argument("--prompt_type", choices=["none", "zeroshot_cot", "manual_cot"], default="manual_cot")
    parser.add_argument("--eval_type", choices=["direct", "re", "json", "llm"], default="re")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--dataset_path", default="dataset")
    parser.add_argument("--result_path", default="result")
    args = parser.parse_args()

    os.makedirs(args.result_path, exist_ok=True)
    output_file = os.path.join(args.result_path, f"{args.model_id}_{args.prompt_type}_{args.eval_type}.json")
    print('output_file: ', output_file)

    if args.resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as fp:
            result = json.load(fp)
        done = set(result.get("Processed", []))
        print(f"[INFO] Resuming - {len(done)} samples already done")
    else:
        result = blank_stats()
        done = set()
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=4)

    with open(os.path.join(args.dataset_path, "data.json"), "r", encoding="utf-8") as fp:
        data = json.load(fp)
        items = []
        for i in range(args.repeat):
            for it in data:
                temp = it.copy()
                temp['repeat'] = i
                items.append(temp)

    todo = []
    for it in items:
        uid = f"{it['task_type']}_{it['id']}_{it['repeat']}"
        if uid not in done:
            todo.append(it)
    print("Total tasks:", len(items), " | remaining:", len(todo))

    client = make_client()
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(process_info, it, args=args, client=client): f"{it['sub_task_type']}_{it['id']}"
            for it in todo
        }
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                ok, task, sub, uid = f.result()
                safe_update_and_dump(result, ok, task, sub, uid, output_file, lock)
            except Exception as e:
                print("[ERROR] worker failed", e)

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
