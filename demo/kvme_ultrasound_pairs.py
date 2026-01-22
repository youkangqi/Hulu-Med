import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hulu-Med-7B on paired ultrasound images from the KVME dataset."
    )
    parser.add_argument(
        "--model-path",
        default="/homeB/youkangqi/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-7B/snapshots/258594714a0d3835eb2c9e4cc165a4242e606d71/",
        help="Local path to the Hulu-Med-7B model snapshot.",
    )
    parser.add_argument(
        "--json-path",
        default="data/kVME_data/data/new_Mammary2_with_labels_pretty.json",
        help="Path to the KVME JSON metadata file.",
    )
    parser.add_argument(
        "--images-root",
        default="data/kVME_data/data/Mammary_report",
        help="Root directory that contains the image files referenced by image_path.",
    )
    parser.add_argument(
        "--keywords-path",
        default="data/kVME_data/data/key_technical_description_words.txt",
        help="Path to the technical keywords list.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/kvme_reports",
        help="Directory to write reports.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype to use for model and pixel values.",
    )
    parser.add_argument(
        "--attn-impl",
        default="flash_attention_2",
        help="Attention implementation (e.g., flash_attention_2 or eager).",
    )
    parser.add_argument(
        "--use-think",
        action="store_true",
        help="Include model reasoning in the output.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of test items to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def iter_entries(data):
    if isinstance(data, list):
        for item in data:
            yield item
        return
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    yield item
        return
    raise ValueError("Unsupported JSON structure.")


def load_keywords(path: Path) -> list[str]:
    if not path.exists():
        print(f"Warning: keywords file not found: {path}")
        return []
    keywords = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        term = line.strip()
        if term:
            keywords.append(term)
    return keywords


def build_prompt(keywords: list[str]) -> str:
    base = (
        "请根据这两张乳腺超声图像生成中文诊断报告，使用结构化输出：\n"
        "【影像所见】\n【印象】\n【BI-RADS】\n【建议】\n"
        "如有不确定之处，请明确说明不确定。"
    )
    if not keywords:
        return base
    return (
        base
        + "\n请尽量对以下专业术语进行描述：\n"
        + "、".join(keywords)
    )


def generate_report(
    model,
    processor,
    tokenizer,
    image_paths: list[Path],
    prompt: str,
    torch_dtype: torch.dtype,
    max_new_tokens: int,
    use_think: bool,
):
    content = [
        {"type": "image", "image": {"image_path": str(image_paths[0])}},
        {"type": "image", "image": {"image_path": str(image_paths[1])}},
        {"type": "text", "text": prompt},
    ]
    conversation = [{"role": "user", "content": content}]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        use_think=use_think,
    )[0].strip()


def main():
    seed = 2026
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args = parse_args()

    json_path = Path(args.json_path)
    images_root = Path(args.images_root)
    keywords_path = Path(args.keywords_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries = [item for item in iter_entries(data) if item.get("split") == "test"]
    if args.max_items is not None:
        entries = entries[: args.max_items]

    torch_dtype = resolve_dtype(args.dtype)
    attn_impl = args.attn_impl
    if not torch.cuda.is_available():
        torch_dtype = torch.float32
        attn_impl = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer

    keywords = load_keywords(keywords_path)
    prompt = build_prompt(keywords)

    for item in entries:
        uid = str(item.get("uid", "unknown"))
        image_names = item.get("image_path", [])
        if not isinstance(image_names, list) or len(image_names) < 2:
            print(f"Skip {uid}: invalid image_path")
            continue

        image_paths = [images_root / image_names[0], images_root / image_names[1]]
        if not image_paths[0].exists() or not image_paths[1].exists():
            print(f"Skip {uid}: missing image files")
            continue

        output_path = output_dir / f"{uid}.txt"
        if output_path.exists() and not args.overwrite:
            print(f"Skip {uid}: output exists")
            continue

        print(f"Processing uid={uid}")
        report = generate_report(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image_paths=image_paths,
            prompt=prompt,
            torch_dtype=torch_dtype,
            max_new_tokens=args.max_new_tokens,
            use_think=args.use_think,
        )

        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"uid: {uid}\n")
            f.write(f"image_path_1: {image_paths[0]}\n")
            f.write(f"image_path_2: {image_paths[1]}\n\n")
            f.write(report)


if __name__ == "__main__":
    main()
