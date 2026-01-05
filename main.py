import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = r"E:\models\Tencent-Hunyuan\HY-MT1___5-1___8B-FP8"


def cuda_test():
    print("torch:", torch.__version__)
    print("cuda build:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def build_translate_prompt(en_text: str) -> str:
    # 通用“翻译指令”prompt；如果模型有更严格的格式要求，再按模型说明调整
    return (
        "你是专业翻译。请把下面英文翻译成中文，只输出译文，不要解释。\n\n"
        f"英文：{en_text}\n"
        "中文："
    )


def translate_en_to_zh(tokenizer, model, en_text: str) -> str:
    prompt = build_translate_prompt(en_text)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)  # 避免 generate 报未使用参数
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # 翻译建议关闭采样，稳定
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # 尽量只取“中文：”后面的部分
    if "中文：" in text:
        text = text.split("中文：", 1)[-1].strip()
    return text


def main():
    cuda_test()
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).eval()

    # 生成时开启 cache 通常更快（即使 config.json 里是 false，也可以运行时覆盖）
    model.config.use_cache = True

    en = "Please translate this sentence into Chinese: I love programming."
    zh = translate_en_to_zh(tokenizer, model, en)
    print("EN:", en)
    print("ZH:", zh)


if __name__ == "__main__":
    main()