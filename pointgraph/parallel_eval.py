import os
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_eval(group_index, group, all_visible_nodes, model_id, repeat, dataset_path, result_path, eval_type):
    visible_nodes_list = all_visible_nodes.split(",")
    eval_per_node =  len(visible_nodes_list) // group
    cuda_visible_index = [
        group_index * eval_per_node + i
        for i in range(eval_per_node)
    ]
    cuda_visible_devices = [visible_nodes_list[i] for i in cuda_visible_index]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

    if "qwen" in model_id.lower():
        script_type = "vlm"
    else:
        script_type = "api"

    cmd = [
        "python",
        f"{script_type}_eval.py",
        "--group_index",
        str(group_index),
        "--group",
        str(group),
        "--repeat",
        str(repeat),
        "--model_id",
        model_id,
        "--dataset_path",
        dataset_path,
        "--result_path",
        result_path,
        "--eval_type",
        eval_type,
    ]
    return subprocess.Popen(cmd, env=env)


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
    parser.add_argument("--dataset_path", default="../dataset")
    parser.add_argument("--result_path", default="result")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--group", type=int, default=8) # 8 parallel processes
    parser.add_argument("--visible_nodes", type=str, default="0,1,2,3,4,5,6,7") # 0,1,2,3,4,5,6,7
    # parallel eval parameters, default is 8 parallel processes, every process will use 1 GPU
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--eval_type", choices=["direct", "re", "json", "llm"], default="re")
    args = parser.parse_args()

    result_path = args.result_path
    dataset_path = args.dataset_path
    os.makedirs(result_path, exist_ok=True)

    # inference
    group = args.group
    visible_nodes = args.visible_nodes

    model_id = args.model_id
    repeat = args.repeat
    eval_type = args.eval_type

    procs = []
    with ThreadPoolExecutor(max_workers=group) as executor:
        futures = [
            executor.submit(run_eval, i, group, visible_nodes, model_id, repeat, dataset_path, result_path, eval_type)
            for i in range(group)
        ]
        procs = [f.result() for f in futures]

    for p in procs:
        p.wait()
    print("Done!")

    # evaluation
    overall_result = blank_stats()
    
    for group_index in range(group):
        result = json.load(open(os.path.join(result_path, model_id, f"results_{group_index}.json")))
        for task in result.keys():
            if task == 'Total':
                overall_result[task].extend(result[task])
            else:
                for sub_task in result[task].keys():
                    overall_result[task][sub_task].extend(result[task][sub_task])

    # final report -----------------------------------------------------------
    eps = 1e-6
    overall = sum(overall_result["Total"]) / (len(overall_result["Total"])+eps) * 100
    print("\n======= FINAL =======")
    print(f"Overall: {overall:.2f}% (N={len(overall_result['Total'])})")
    for task in [k for k in overall_result if k not in {"Total", "Processed"}]:
        task_acc = sum(overall_result[task]["Total"]) / (len(overall_result[task]["Total"])+eps) * 100
        print(f"{task}: {task_acc:.2f}%")
        for sub in overall_result[task]:
            if sub == "Total":
                continue
            sub_acc = sum(overall_result[task][sub]) / (len(overall_result[task][sub])+eps) * 100
            print(f"    {sub}: {sub_acc:.2f}%")

    with open(os.path.join(result_path, model_id, "overall_result.json"), "w") as f:
        json.dump(overall_result, f, indent=4)
