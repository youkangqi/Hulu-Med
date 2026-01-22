import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="从一张2D超声图片中生成一份医学影像报告"
    )
    parser.add_argument(
        "--model-path",
        default="/homeB/youkangqi/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-7B/snapshots/258594714a0d3835eb2c9e4cc165a4242e606d71/",
        help="Local path to the Hulu-Med-7B model snapshot.",
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to the 2D breast ultrasound image.",
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
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def main():
    seed = 2026
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args = parse_args()

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

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": {"image_path": args.image_path},
                },
                {
                    "type": "text",
                    "text": (
                        "针对这张二维乳腺图像生成一份结构化的诊断/病理报告。"#"Generate a structured diagnostic/pathology-style report for this 2D breast "#
                        "超声图像。使用部分：检查结果、印象、BI-RADS分级、建议。"#"ultrasound image. Use sections: Findings, Impression, BI-RADS, Recommendations. "#
                        "如有疑问，请明确说明不确定性。"#"If uncertain, explicitly state uncertainty."#
                    ),
                },
            ],
        }
    ]

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
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    outputs = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        use_think=args.use_think,
    )[0].strip()
    print(outputs)


if __name__ == "__main__":
    main()
