import argparse
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select one labeled and one unlabeled 2D ultrasound image per patient, "
            "run Hulu-Med-7B inference, and save reports."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/homeB/youkangqi/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-7B/snapshots/258594714a0d3835eb2c9e4cc165a4242e606d71/",
        help="Local path to the Hulu-Med-7B model snapshot.",
    )
    parser.add_argument(
        "--data-root",
        default="data/abus_data",
        help="Root directory containing trainimg/ and trainlabel/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write prediction outputs.",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Only process a single patient ID.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit number of patients to process.",
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
        default="data/kVME_data/data/key_technical_words.txt",
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
        "--random-select",
        action="store_true",
        help="Randomly pick one labeled and one unlabeled image per patient.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random-select.",
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
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def safe_filename(name: str) -> str:
    return name.replace(" ", "_")


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
            "Please generate a structured diagnostic/pathology-style report for this "
            "2D breast ultrasound image using sections:\n"
            "Findings\nImpression\nBI-RADS\nRecommendations\n"
            "If uncertain, explicitly state the uncertainty."
        )
        base = ("Generate a medical report for this image.")
    else:
        base = (
            "请根据这张2D乳腺超声图像生成中文诊断/病理风格报告，使用结构化输出：\n"
            "【影像所见】\n【印象】\n【BI-RADS】\n【建议】\n"
            "如有不确定之处，请明确说明不确定。"
        )
        base = ("根据这个图片生成中文医学报告")
    if not keywords:
        return base
    if language == "en":
        return base + "\nPlease try to use the following technical terms:\n" + ", ".join(keywords)
    return base + "\n请尽量使用以下专业术语进行描述：\n" + "、".join(keywords)


def generate_report(
    model,
    processor,
    tokenizer,
    image_path: Path,
    prompt: str,
    torch_dtype: torch.dtype,
    max_new_tokens: int,
    use_think: bool,
):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": str(image_path)}},
                {"type": "text", "text": prompt},
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


def write_output(
    output_dir: Path,
    image_path: Path,
    label_present: bool,
    text: str,
    metadata: list[tuple[str, str]] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "with_label" if label_present else "no_label"
    out_name = f"{tag}__{safe_filename(image_path.name)}.txt"
    out_path = output_dir / out_name
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"label_present: {str(label_present).lower()}\n")
        f.write(f"image_path: {image_path}\n\n")
        if metadata:
            for key, value in metadata:
                f.write(f"{key}: {value}\n")
            f.write("\n")
        f.write(text)
    return out_path


def main():
    seed = 2026
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    args = parse_args()

    data_root = Path(args.data_root)
    trainimg = data_root / "trainimg"
    trainlabel = data_root / "trainlabel"
    if not trainimg.exists():
        raise FileNotFoundError(f"Missing trainimg directory: {trainimg}")

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

    patient_dirs = sorted([p for p in trainimg.iterdir() if p.is_dir()])
    if args.patient_id:
        patient_dirs = [trainimg / args.patient_id]

    rng = None
    selection_mode = "first"
    seed_value = "none"
    if args.random_select:
        rng = random.Random(args.seed)
        selection_mode = "random"
        seed_value = str(args.seed) if args.seed is not None else "auto"

    processed = 0
    for patient_dir in patient_dirs:
        if not patient_dir.exists():
            print(f"Skip missing patient folder: {patient_dir}")
            continue

        label_dir = trainlabel / patient_dir.name
        label_names = {p.name for p in list_images(label_dir)}
        images = list_images(patient_dir)
        labeled_images = [p for p in images if p.name in label_names]
        unlabeled_images = [p for p in images if p.name not in label_names]

        if not labeled_images or not unlabeled_images:
            print(f"Skip {patient_dir.name}: need at least one labeled and one unlabeled image.")
            continue

        if rng is None:
            labeled_image = labeled_images[0]
            unlabeled_image = unlabeled_images[0]
        else:
            labeled_image = rng.choice(labeled_images)
            unlabeled_image = rng.choice(unlabeled_images)

        labeled_index = images.index(labeled_image)
        unlabeled_index = images.index(unlabeled_image)

        output_dir = Path(args.output_dir) / patient_dir.name
        labeled_out = output_dir / f"with_label__{safe_filename(labeled_image.name)}.txt"
        unlabeled_out = output_dir / f"no_label__{safe_filename(unlabeled_image.name)}.txt"

        if not args.overwrite and labeled_out.exists() and unlabeled_out.exists():
            print(f"Skip {patient_dir.name}: outputs already exist.")
            continue

        print(f"Processing patient {patient_dir.name}")

        labeled_text = generate_report(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image_path=labeled_image,
            prompt=prompt,
            torch_dtype=torch_dtype,
            max_new_tokens=args.max_new_tokens,
            use_think=args.use_think,
        )
        labeled_metadata = [
            ("selection_mode", selection_mode),
            ("selection_seed", seed_value),
            ("index_in_patient_sequence_0based", str(labeled_index)),
            ("index_in_patient_sequence_1based", str(labeled_index + 1)),
            ("total_images_in_patient", str(len(images))),
        ]
        write_output(output_dir, labeled_image, True, labeled_text, labeled_metadata)

        unlabeled_text = generate_report(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image_path=unlabeled_image,
            prompt=prompt,
            torch_dtype=torch_dtype,
            max_new_tokens=args.max_new_tokens,
            use_think=args.use_think,
        )
        unlabeled_metadata = [
            ("selection_mode", selection_mode),
            ("selection_seed", seed_value),
            ("index_in_patient_sequence_0based", str(unlabeled_index)),
            ("index_in_patient_sequence_1based", str(unlabeled_index + 1)),
            ("total_images_in_patient", str(len(images))),
        ]
        write_output(output_dir, unlabeled_image, False, unlabeled_text, unlabeled_metadata)

        processed += 1
        if args.max_patients is not None and processed >= args.max_patients:
            break


if __name__ == "__main__":
    main()
