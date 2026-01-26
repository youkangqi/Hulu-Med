import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


class SafeDict(dict):
    def __missing__(self, key):
        return ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run text-only inference on rows from outputs/abus_b_g.csv."
    )
    parser.add_argument(
        "--model-path",
        default="/homeB/youkangqi/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-7B/snapshots/258594714a0d3835eb2c9e4cc165a4242e606d71/",
        help="Local path to the Hulu-Med-7B model snapshot.",
    )
    parser.add_argument(
        "--csv-path",
        default="outputs/abus_b_g.csv",
        help="Path to the CSV file with columns: row,col_B,col_G.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/abus_b_g_infer",
        help="Directory to write model outputs.",
    )
    parser.add_argument(
        "--prompt-template",
        default=(
            "以下是超声检查记录：\n"
            "超声号：{col_B}\n"
            "分类：{col_G}\n"
            "根据文本描述，解释临床含义，直接给出BI-RADS分级，使用结构化输出：\n"
            "【临床含义】\n【BI-RADS】\n【医疗建议】\n"
        ),
        help="Prompt template with placeholders like {col_B}, {col_G}, {row}.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype to use for model.",
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
        help="Limit number of rows to process.",
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


def safe_filename(value: str) -> str:
    cleaned = value.strip().replace(" ", "_")
    return cleaned.replace("/", "_").replace("\\", "_")


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("col_B"):
                continue

            prompt = args.prompt_template.format_map(SafeDict(row))
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

            inputs = processor(
                conversation=conversation,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            ultrasound_id = safe_filename(str(row.get("col_B", "")))
            if not ultrasound_id:
                ultrasound_id = str(row.get("row", processed + 1))
            output_path = output_dir / f"{ultrasound_id}.txt"
            if output_path.exists() and not args.overwrite:
                processed += 1
                if args.max_items is not None and processed >= args.max_items:
                    break
                continue

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.6,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            output_text = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                use_think=args.use_think,
            )[0].strip()

            with output_path.open("w", encoding="utf-8") as out:
                out.write(f"row: {row.get('row', '')}\n")
                out.write(f"col_B: {row.get('col_B', '')}\n")
                out.write(f"col_G: {row.get('col_G', '')}\n\n")
                out.write(output_text)

            processed += 1
            if args.max_items is not None and processed >= args.max_items:
                break

    print(f"Processed {processed} rows. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
