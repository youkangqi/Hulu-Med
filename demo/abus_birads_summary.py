import argparse
from pathlib import Path
import re

'''
会遍历 input_dir 下每个病人目录，从以下文件提取 BI‑RADS：

labelled_infer.txt（多张图，输出包含 image_index）
unlabelled_infer.txt（多张图，输出包含 image_index）
multi_images_report.txt
multi_images_report_labels_only.txt

提取结构化报告中BI-RADS等级,输出 CSV 到 output-csv
python demo/abus_birads_summary.py --input-dir outputs/abus_reports_cn_with_keys --output-csv outputs/abus_reports_cn_with_keys/birads_summary.csv
'''

IMAGE_EXT_RE = re.compile(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", re.I)
BI_RADS_RE = re.compile(
    r"BI\s*[-]?\s*RADS[^0-9A-Za-z]{0,40}(?:Category\s*)?([0-6](?:[A-Ca-c])?)",
    re.I,
)
SECTION_STOP_RE = re.compile(r"^【|^Findings:|^Impression:|^BI", re.I)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize BI-RADS ratings from ABUS report outputs."
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/abus_reports_cn",
        help="Root directory containing per-patient folders.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/abus_reports_cn/birads_summary.csv",
        help="CSV path to write the summary.",
    )
    return parser.parse_args()


def extract_birads(text: str) -> str:
    match = BI_RADS_RE.search(text)
    if not match:
        return ""
    return match.group(1).upper()


def extract_section(text: str, keys: list[str]) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    capture = False
    buf = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(k) for k in keys):
            capture = True
            continue
        if capture and SECTION_STOP_RE.match(stripped):
            break
        if capture:
            buf.append(line)
    return "\n".join(buf).strip()


def parse_aggregate_file(path: Path, patient_id: str, source: str):
    records = []
    if not path.exists():
        return records

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current = None

    def flush_current():
        if not current:
            return
        text = "\n".join(current["text"])
        findings = extract_section(text, ["【影像所见】", "影像所见", "Findings:"])
        impression = extract_section(text, ["【印象】", "印象", "Impression:"])
        records.append(
            {
                "patient_id": patient_id,
                "source": source,
                "image_index": current.get("image_index", ""),
                "image_path": current.get("image_path", ""),
                "birads": extract_birads(text),
                "findings": findings,
                "impression": impression,
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
    text = path.read_text(encoding="utf-8", errors="ignore")
    findings = extract_section(text, ["【影像所见】", "影像所见", "Findings:"])
    impression = extract_section(text, ["【印象】", "印象", "Impression:"])
    return [
        {
            "patient_id": patient_id,
            "source": source,
            "image_index": "",
            "image_path": "",
            "birads": extract_birads(text),
            "findings": findings,
            "impression": impression,
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
    with output_csv.open("w", encoding="utf-8") as f:
        f.write("patient_id,source,image_index,image_path,birads,findings,impression\n")
        for row in rows:
            findings = row.get("findings", "").replace("\n", "\\n")
            impression = row.get("impression", "").replace("\n", "\\n")
            f.write(
                f"{row['patient_id']},{row['source']},"
                f"{row['image_index']},{row['image_path']},{row['birads']},"
                f"{findings},{impression}\n"
            )

    print(f"Wrote {output_csv} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
