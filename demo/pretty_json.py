import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretty-print a JSON file without changing its data structure."
    )
    parser.add_argument(
        "input_path",
        help="Path to the JSON file to format.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write the formatted JSON. Defaults to <input>.pretty.json.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file in place.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent size for pretty JSON.",
    )
    parser.add_argument(
        "--ensure-ascii",
        action="store_true",
        help="Escape non-ASCII characters in output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # utf-8-sig strips BOM if present.
    with input_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if args.in_place:
        output_path = input_path
    else:
        output_path = Path(args.output_path) if args.output_path else input_path.with_suffix(
            input_path.suffix + ".pretty.json"
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=args.ensure_ascii,
            indent=args.indent,
            sort_keys=False,
        )
        f.write("\n")

    print(f"Wrote formatted JSON to {output_path}")


if __name__ == "__main__":
    main()
