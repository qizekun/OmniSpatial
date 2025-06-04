import os
import io
import time
import base64
from PIL import Image
from openai import OpenAI
from serve.system_prompts import LLM_JUDGE_SYSTEM_PROMPT

###############################################################################
#                             OpenAI client setup                             #
###############################################################################

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/")

def make_client(api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_API_BASE):
    """Construct an OpenAI client (proxy-friendly)."""
    return OpenAI(api_key=api_key, base_url=base_url)


###############################################################################
#                          PIL <-> base64 utilities                           #
###############################################################################

def encode_pil_image(img: Image.Image) -> str:
    fmt = "PNG" if img.mode == "RGBA" else "JPEG"
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


###############################################################################
#                        Core chat / retry primitives                         #
###############################################################################

def _chat_with_retry(messages, model: str, client: OpenAI, *, tries: int = 10):
    for attempt in range(tries):
        try:
            comp = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=2000,
            )
            return comp.choices[0].message.content
        except Exception as e:
            if attempt == tries - 1:
                print("[FATAL] OpenAI error", e)
                return "A"  # fallback
            print("[WARN] OpenAI error - retrying", e)
            time.sleep(1 + attempt)


###############################################################################
#                      Vision-language question answering                     #
###############################################################################

def vqa_parsing(prompt: str, img: Image.Image, *, sys_prompt: str, model: str, client: OpenAI):
    base64_img = encode_pil_image(img)
    msgs = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"}},
            ],
        },
    ]
    return _chat_with_retry(msgs, model, client)


###############################################################################
#                            LLM correctness judge                            #
###############################################################################

def llm_judge(question: str, pred: str, gt: str, client: OpenAI, *, judge_model: str = "gpt-4.1-nano") -> bool:
    msgs = [
        {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": f"Question: {question}\nPred: {pred}\nGT: {gt}"}]},
    ]
    res = _chat_with_retry(msgs, judge_model, client)
    return "true" in res.lower()
