import os
import threading
from typing import Iterator, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

_tokenizer: Optional[object] = None
_model: Optional[object] = None


def init(model_dir: str) -> Tuple[object, object]:
    """加载 tokenizer/model（建议在 FastAPI startup 调用一次）"""
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(model_dir)

    _tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).eval()

    # 生成更快
    _model.config.use_cache = True
    return _tokenizer, _model


def cuda_info() -> dict:
    return {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def build_translate_prompt(en_text: str) -> str:
    return (
        "你是专业翻译。请把下面英文翻译成中文，只输出译文，不要解释。\n\n"
        f"英文：{en_text}\n"
        "中文："
    )


def _prepare_inputs(prompt: str):
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not initialized. Call infer.init(MODEL_DIR) first.")

    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)  # 避免 generate 报未使用参数
    return {k: v.to(_model.device) for k, v in inputs.items()}


def translate(en_text: str, max_new_tokens: int = 256) -> str:
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not initialized. Call infer.init(MODEL_DIR) first.")

    prompt = build_translate_prompt(en_text)
    inputs = _prepare_inputs(prompt)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if "中文：" in text:
        text = text.split("中文：", 1)[-1].strip()
    return text


def stream_translate(en_text: str, max_new_tokens: int = 256) -> Iterator[str]:
    """流式产出文本分片（不负责 SSE 包装）"""
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not initialized. Call infer.init(MODEL_DIR) first.")

    prompt = build_translate_prompt(en_text)
    inputs = _prepare_inputs(prompt)

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    def _run():
        with torch.no_grad():
            _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    for piece in streamer:
        yield piece