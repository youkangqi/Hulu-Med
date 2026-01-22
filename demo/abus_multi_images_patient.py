import argparse
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hulu-Med-7B on a full 2D ultrasound image sequence for one patient."
    )
    parser.add_argument(
        "--model-path",
        default="/homeB/youkangqi/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-7B/snapshots/258594714a0d3835eb2c9e4cc165a4242e606d71/",
        help="Local path to the Hulu-Med-7B model snapshot.",
    )
    parser.add_argument(
        "--data-root",
        default="data/abus_data",
        help="Root directory containing trainimg/ folder.",
    )
    parser.add_argument(
        "--patient-id",
        required=False,
        help="Patient ID folder under trainimg/ (e.g., 33062).",
    )
    parser.add_argument(
        "--all-patients",
        action="store_true",
        help="Process all patients under trainimg/ (overrides --patient-id).",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only use trainimg images whose filenames appear in trainlabel/.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path to save the generated report. Defaults to outputs/{patient_id}/multi_images_report.txt",
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
        "--keywords-path",
        default="data/kVME_data/data/key_technical_description_words_v1.txt",
        help="Path to the technical keywords list.",
    )
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default="zh",
        help="Prompt language (zh or en).",
    )
    parser.add_argument(
        "--use-think",
        action="store_true",
        help="Include model reasoning in the output.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the prompt text and its length, then continue.",
    )
    parser.add_argument(
        "--print-input-length",
        action="store_true",
        help="Print total input token length per patient before generation.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Subsample images by stride (1 keeps all images).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images to include after stride.",
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


def list_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


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


def build_prompt(keywords: list[str], language: str) -> str:
    if language == "en":
        base = (
            "This is a sequence of 2D breast ultrasound images from a single patient. "
            "Generate a structured diagnostic/pathology-style report for the sequence "
            "using sections:\n"
            "Findings\nImpression\nBI-RADS\nRecommendations\n"
            "If uncertain, explicitly state the uncertainty."
        )
    else:
        base = (
            "这是一位患者的2D乳腺超声序列图像。请基于整段序列生成中文诊断/病理风格报告，"
            "使用结构化输出：\n"
            "【影像所见】\n【印象】\n【BI-RADS】\n【建议】\n"
            "如有不确定之处，请明确说明不确定。"
        )
        base = ("这是一位患者的2D乳腺超声序列图像，请基于整段序列生成中文医学报告")
    if not keywords:
        return base
    if language == "en":
        return base + "\nPlease try to use the following technical terms:\n" + ", ".join(keywords)
    return base + "\n请尽量使用以下专业术语进行描述：\n" + "、".join(keywords)


def resolve_output_path(args, patient_dir: Path) -> Path:
    if args.output_path:
        base = Path(args.output_path)
        is_dir_like = args.all_patients or base.is_dir() or base.suffix == ""
        if is_dir_like:
            filename = (
                "multi_images_report_labels_only.txt"
                if args.labels_only
                else "multi_images_report.txt"
            )
            return base / patient_dir.name / filename
        return base
    if args.labels_only:
        return Path("outputs") / patient_dir.name / "multi_images_report_labels_only.txt"
    return Path("outputs") / patient_dir.name / "multi_images_report.txt"


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
    tokenizer = processor.tokenizer
    keywords = load_keywords(Path(args.keywords_path))
    prompt = build_prompt(keywords, args.language)
    if args.print_prompt:
        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]
        print("=== PROMPT TEXT START ===")
        print(prompt)
        print("=== PROMPT TEXT END ===")
        print(f"Prompt chars: {len(prompt)}")
        print(f"Prompt tokens: {len(prompt_tokens)}")

    data_root = Path(args.data_root)
    trainimg_root = data_root / "trainimg"
    trainlabel_root = data_root / "trainlabel"
    if args.all_patients:
        patient_dirs = sorted([p for p in trainimg_root.iterdir() if p.is_dir()])
    else:
        if not args.patient_id:
            raise ValueError("Either --patient-id or --all-patients must be provided.")
        patient_dirs = [trainimg_root / args.patient_id]

    for patient_dir in patient_dirs:
        if not patient_dir.exists():
            print(f"Skip missing patient folder: {patient_dir}")
            continue

        images = list_images(patient_dir)
        if args.labels_only:
            label_dir = trainlabel_root / patient_dir.name
            label_names = [p.name for p in list_images(label_dir)]
            images = [patient_dir / name for name in label_names if (patient_dir / name).exists()]
        if not images:
            print(f"Skip empty folder: {patient_dir}")
            continue

        if args.stride > 1:
            images = images[:: args.stride]
        if args.max_images is not None:
            images = images[: args.max_images]

        content = [{"type": "image", "image": {"image_path": str(p)}} for p in images]
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]

        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if args.print_input_length and "input_ids" in inputs:
            print(f"Input keys for {patient_dir.name}: {list(inputs.keys())}")
            print(f"Input tokens for {patient_dir.name}: {inputs['input_ids'].shape[-1]}")
            if "pixel_values" in inputs:
                print(f"pixel_values shape for {patient_dir.name}: {tuple(inputs['pixel_values'].shape)}")
            if "image_grid_thw" in inputs:
                grid = inputs["image_grid_thw"]
                grid_shape = tuple(grid.shape) if hasattr(grid, "shape") else ()
                grid_count = grid_shape[0] if grid_shape else "unknown"
                print(f"image_grid_thw shape for {patient_dir.name}: {grid_shape} (count={grid_count})")
            if "image_sizes" in inputs:
                sizes = inputs["image_sizes"]
                if hasattr(sizes, "shape"):
                    size_count = sizes.shape[0]
                    size_shape = tuple(sizes.shape)
                else:
                    size_count = len(sizes)
                    size_shape = "list"
                print(f"image_sizes for {patient_dir.name}: {size_shape} (count={size_count})")
            if "grid_sizes" in inputs:
                grid_sizes = inputs["grid_sizes"]
                if hasattr(grid_sizes, "shape"):
                    grid_shape = tuple(grid_sizes.shape)
                    grid_count = grid_sizes.shape[0]
                else:
                    grid_shape = "list"
                    grid_count = len(grid_sizes)
                print(f"grid_sizes for {patient_dir.name}: {grid_shape} (count={grid_count})")
                if hasattr(grid_sizes, "tolist"):
                    grid_list = grid_sizes.tolist()
                else:
                    grid_list = grid_sizes
                for i, dims in enumerate(grid_list):
                    if len(dims) >= 3:
                        t, h, w = dims[0], dims[1], dims[2]
                        tokens = t * h * w
                        print(f"grid_sizes[{i}] = {dims} -> tokens={tokens}")
                    elif len(dims) == 2:
                        h, w = dims[0], dims[1]
                        tokens = h * w
                        print(f"grid_sizes[{i}] = {dims} -> tokens={tokens}")
            if "merge_sizes" in inputs:
                merge_sizes = inputs["merge_sizes"]
                if hasattr(merge_sizes, "shape"):
                    merge_shape = tuple(merge_sizes.shape)
                    merge_count = merge_sizes.shape[0]
                else:
                    merge_shape = "list"
                    merge_count = len(merge_sizes)
                print(f"merge_sizes for {patient_dir.name}: {merge_shape} (count={merge_count})")
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_text = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            use_think=args.use_think,
        )[0].strip()

        output_path = resolve_output_path(args, patient_dir)
        if output_path.exists() and not args.overwrite:
            print(f"Skip {patient_dir.name}: output exists.")
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")

        print(f"Processed {len(images)} images for patient {patient_dir.name}.")
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
