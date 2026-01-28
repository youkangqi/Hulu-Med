import argparse
import csv
from pathlib import Path
import re

'''
会遍历 outputs/abus_reports_raw_report_cn 下每个病人目录，提取：

labelled_infer.txt（多张图：保存 image_index + 输出文本）
unlabelled_infer.txt（多张图：保存 image_index + 输出文本）
multi_images_report.txt
multi_images_report_labels_only.txt

输出为 CSV（默认）：
outputs_extracted.csv

使用方法：
python demo/abus_outputs_extract.py --input-dir outputs/abus_reports_raw_report_cn --output-csv outputs/abus_reports_raw_report_cn/outputs_extracted.csv

'''

IMAGE_EXT_RE = re.compile(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", re.I)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract model outputs from ABUS report folders."
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/abus_reports_raw_report_cn",
        help="Root directory containing per-patient folders.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/abus_reports_raw_report_cn/outputs_extracted.csv",
        help="CSV path to write extracted outputs.",
    )
    return parser.parse_args()


def parse_aggregate_file(path: Path, patient_id: str, source: str):
    records = []
    if not path.exists():
        return records

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current = None

    def flush_current():
        if not current:
            return
        text = "\n".join(current["text"]).strip()
        records.append(
            {
                "patient_id": patient_id,
                "source": source,
                "image_index": current.get("image_index", ""),
                "image_path": current.get("image_path", ""),
                "output_text": text,
            }
        )

    for line in lines:
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2 and IMAGE_EXT_RE.search(parts[0]):
                flush_current()
                current = {
                    "image_path": parts[0].strip(),
                    "image_index": parts[1].strip(),
                    "text": [],
                }
                continue
        if current is not None:
            current["text"].append(line)

    flush_current()
    return records


def parse_single_report(path: Path, patient_id: str, source: str):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [
        {
            "patient_id": patient_id,
            "source": source,
            "image_index": "",
            "image_path": "",
            "output_text": text,
        }
    ]


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rows = []
    for patient_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        pid = patient_dir.name
        rows.extend(
            parse_aggregate_file(patient_dir / "labelled_infer.txt", pid, "labelled_infer")
        )
        rows.extend(
            parse_aggregate_file(patient_dir / "unlabelled_infer.txt", pid, "unlabelled_infer")
        )
        rows.extend(
            parse_single_report(
                patient_dir / "multi_images_report.txt", pid, "multi_images_report"
            )
        )
        rows.extend(
            parse_single_report(
                patient_dir / "multi_images_report_labels_only.txt",
                pid,
                "multi_images_report_labels_only",
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "source", "image_index", "image_path", "output_text"])
        for row in rows:
            writer.writerow(
                [
                    row["patient_id"],
                    row["source"],
                    row["image_index"],
                    row["image_path"],
                    row["output_text"],
                ]
            )

    print(f"Wrote {output_csv} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
