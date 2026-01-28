import argparse
import csv
from pathlib import Path

'''
会在 birads_summary.csv 里按 patient_id 去 abus_b_g.csv 的 B 列（col_B）匹配，并把 C 列（col_G）写入指定列。
默认输出到：
outputs/abus_reports_cn/birads_summary_with_colG.csv

使用方法：
python demo/merge_birads_with_abus_bg.py --birads-csv outputs/abus_reports_cn/birads_summary.csv \
--output-csv outputs/abus_reports_cn/birads_summary_with_colG.csv


如果要直接覆盖原文件：
python demo/merge_birads_with_abus_bg.py --in-place

也可以自定义目标列名（比如 category）：
python demo/merge_birads_with_abus_bg.py --target-column category

'''

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge outputs/abus_b_g.csv column C (col_G) into birads_summary.csv "
            "by matching patient_id to col_B."
        )
    )
    parser.add_argument(
        "--birads-csv",
        default="outputs/abus_reports_cn/birads_summary.csv",
        help="Path to birads_summary.csv.",
    )
    parser.add_argument(
        "--abus-csv",
        default="outputs/abus_b_g.csv",
        help="Path to abus_b_g.csv.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/abus_reports_cn/birads_summary_with_colG.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--birads-key",
        default="patient_id",
        help="Column name in birads CSV for patient id.",
    )
    parser.add_argument(
        "--abus-key",
        default="col_B",
        help="Column name in abus CSV for patient id (B column).",
    )
    parser.add_argument(
        "--abus-value",
        default="col_G",
        help="Column name in abus CSV for value to copy (C column).",
    )
    parser.add_argument(
        "--target-column",
        default="birads_gt",
        help="Column name to write into birads CSV.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite birads CSV in place.",
    )
    return parser.parse_args()


def normalize(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def main():
    args = parse_args()
    birads_csv = Path(args.birads_csv)
    abus_csv = Path(args.abus_csv)
    output_csv = Path(args.output_csv)

    if not birads_csv.exists():
        raise FileNotFoundError(f"Birads CSV not found: {birads_csv}")
    if not abus_csv.exists():
        raise FileNotFoundError(f"Abus CSV not found: {abus_csv}")

    # Build mapping from col_B -> col_G
    mapping = {}
    conflicts = 0
    with abus_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize(row.get(args.abus_key))
            value = normalize(row.get(args.abus_value))
            if not key:
                continue
            if key in mapping and value and mapping[key] != value:
                conflicts += 1
                continue
            if value:
                mapping[key] = value

    out_path = birads_csv if args.in_place else output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with birads_csv.open("r", encoding="utf-8", newline="") as f_in, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if args.target_column not in fieldnames:
            fieldnames.append(args.target_column)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            key = normalize(row.get(args.birads_key))
            row[args.target_column] = mapping.get(key, "")
            writer.writerow(row)

    if conflicts:
        print(f"Warning: {conflicts} conflicting IDs in {abus_csv} (kept first value).")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
