import os
import re
import json
import torch
import rembg
import argparse
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.infer_util import remove_background, resize_foreground
from src.utils.camera_util import FOV_to_intrinsics, get_circular_camera_poses
from utils import SYS_PROMPTS, FORMAT_PROMPTS, vqa_parsing_with_scot, llm_judge, make_client

def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames


def blank_stats():
    return {
        "Processed": [], # for resume
        "Total": [],
        "Dynamic_Reasoning": {"Manipulation": [], "Motion_Analysis": [], "Total": []},
        "Spatial_Interaction": {"Traffic_Analysis": [], "Localization": [], "Geospatial_Strategy": [], "Total": []},
        "Complex_Logic": {"Pattern_Recognition": [], "Geometric_Reasoning": [], "Total": []},
        "Perspective_Taking": {"Egocentric": [], "Allocentric": [], "Hypothetical": [], "Total": []},
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
    parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
    parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
    parser.add_argument("--model_id", default="gpt-4.1")
    parser.add_argument("--prompt_type", choices=["none", "zeroshot_cot", "manual_cot"], default="manual_cot")
    parser.add_argument("--eval_type", choices=["direct", "re", "json", "llm"], default="re")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dataset_path", default="../dataset")
    parser.add_argument("--result_path", default="result")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    prompt_type = args.prompt_type
    eval_type = args.eval_type
    dataset_path = args.dataset_path
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    output_file = os.path.join(result_path, f"{model_id}_{prompt_type}_{eval_type}.json")
    print('output_file: ', output_file)

    # ------------------------------------------------------------
    # 1. load model
    # ------------------------------------------------------------

    config = OmegaConf.load(args.config)
    config_name = os.path.basename(args.config).replace('.yaml', '')
    infer_config = config.infer_config

    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

    device = torch.device('cuda')

    # load diffusion model
    print('Loading diffusion model ...')
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    # load custom white-background UNet
    print('Loading custom white-background unet ...')
    if os.path.exists(infer_config.unet_path):
        unet_ckpt_path = infer_config.unet_path
    else:
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)
    pipeline = pipeline.to(device)

    rembg_session = None if args.no_rembg else rembg.new_session()

    # ------------------------------------------------------------
    # 2. start evaluation
    # ------------------------------------------------------------

    client = make_client()

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


    with open(os.path.join(dataset_path, "data.json"), "r", encoding="utf-8") as fp:
        data = json.load(fp)
        data = [sample for sample in data if sample["task_type"] == "Perspective_Taking"]
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

    for it in tqdm(todo):
        raw_id = it["id"]  # e.g. "15_1"
        question = it["question"]
        options = it["options"]
        gt_idx = it["answer"]
        task_type = it["task_type"]
        sub_task = it["sub_task_type"]
        repeat = it["repeat"]
        uid = f"{task_type}_{raw_id}_{repeat}"

        sys_prompt = SYS_PROMPTS[prompt_type] + '\n' + FORMAT_PROMPTS[eval_type]
        prompt = question + "\n"
        for i in range(len(options)):
            prompt += f"{chr(65 + i)}. {options[i]}\n"

        img_path = os.path.join(dataset_path, task_type, f"{raw_id.split('_')[0]}.png")
        img = Image.open(img_path).convert("RGB")

        # # remove background optionally
        # if not args.no_rembg:
        #     img = remove_background(img, rembg_session)
        #     img = resize_foreground(img, 0.85)
        
        # sampling
        multi_view_img = pipeline(
            img, 
            num_inference_steps=args.diffusion_steps, 
        ).images[0]

        response = vqa_parsing_with_scot(prompt, img, novel_view_img=multi_view_img, sys_prompt=sys_prompt, model=model_id, client=client)
        print(response)

        # ---------------------- evaluation -------------------------------
        gt_letter = chr(65 + gt_idx)
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
        
        result["Processed"].append(uid)
        result["Total"].append(flag)
        result[task_type][sub_task].append(flag)
        result[task_type]["Total"].append(flag)
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=4)

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
