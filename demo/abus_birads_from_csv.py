import argparse
import csv
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

'''
读取 outputs_extracted.csv 的 E列（output_text），调用 Hulu‑Med 让模型判断 BI‑RADS，并把结果写到 （birads_pred）列 同一行，输出到新 CSV。

默认输出：outputs/abus_reports_raw_report_cn/outputs_extracted_with_birads.csv

可选参数
--input-csv 指定输入文件
--output-csv 指定输出文件
--text-column 指定文本列名（默认 output_text）
--max-items 只处理前 N 行
--overwrite 覆盖已有输出文件

运行示例
CUDA_VISIBLE_DEVICES=7 python demo/abus_birads_from_csv.py --input-csv outputs/abus_reports_raw_report_cn_with_keys/outputs_extracted.csv --output-csv outputs/abus_reports_raw_report_cn_with_keys/outputs_extracted_with_birads.csv
'''

BIRADS_RE = re.compile(r"([0-6](?:[A-Ca-c])?)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use Hulu-Med to infer BI-RADS from output_text column in CSV."
    )
    parser.add_argument(
        "--model-path",
        default="/homeB/youkangqi/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-7B/snapshots/258594714a0d3835eb2c9e4cc165a4242e606d71/",
        help="Local path to the Hulu-Med-7B model snapshot.",
    )
    parser.add_argument(
        "--input-csv",
        default="outputs/abus_reports_raw_report_cn/outputs_extracted.csv",
        help="Input CSV with output_text in column E.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/abus_reports_raw_report_cn/outputs_extracted_with_birads.csv",
        help="Output CSV with predicted BI-RADS in column F.",
    )
    parser.add_argument(
        "--text-column",
        default="output_text",
        help="Column name to use as input text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
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
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it exists.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of rows to process.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def build_prompt(text: str) -> str:
    return (
        "请根据以下文本判断BI-RADS等级。仅输出等级。"
        "如果无法判断，输出Unknown。\n\n"
        f"{text}"
    )


def extract_birads(text: str) -> str:
    match = BIRADS_RE.search(text)
    if not match:
        return ""
    return match.group(1).upper()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output CSV exists: {output_csv}")

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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    with input_csv.open("r", encoding="utf-8", newline="") as f_in, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if "birads_pred" not in fieldnames:
            fieldnames.append("birads_pred")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            text = row.get(args.text_column, "") or ""
            prompt = build_prompt(text)
            conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            inputs = processor(
                conversation=conversation,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            output_text = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                use_think=False,
            )[0].strip()

            row["birads_pred"] = extract_birads(output_text) or "Unknown"
            writer.writerow(row)

            processed += 1
            if args.max_items is not None and processed >= args.max_items:
                break

    print(f"Wrote {output_csv} with {processed} rows.")


if __name__ == "__main__":
    main()
