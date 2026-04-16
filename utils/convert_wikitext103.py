from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE_DIR = Path("/data/qyq/Wikitext/wikitext-103-v1")
DEFAULT_OUTPUT_DIR = Path("/home/qyq/gpt2_compression/data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local WikiText-103 parquet shards into plain-text files."
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keep_empty_lines",
        action="store_true",
        help="Keep empty rows from the parquet data instead of dropping them.",
    )
    parser.add_argument(
        "--strip_lines",
        action="store_true",
        help="Strip leading and trailing whitespace from each retained line.",
    )
    return parser.parse_args()


def convert_split(
    split_name: str,
    source_dir: Path,
    output_path: Path,
    keep_empty_lines: bool,
    strip_lines: bool,
) -> dict[str, int]:
    shard_paths = sorted(source_dir.glob(f"{split_name}-*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No parquet shards found for split '{split_name}' under {source_dir}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    kept_rows = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for shard_path in shard_paths:
            frame = pd.read_parquet(shard_path, columns=["text"])
            for value in frame["text"].tolist():
                input_rows += 1
                text = "" if value is None else str(value)
                text = text.replace("\r\n", "\n").replace("\r", "\n")

                for line in text.splitlines():
                    candidate = line.strip() if strip_lines else line
                    if not keep_empty_lines and not candidate.strip():
                        continue
                    writer.write(candidate + "\n")
                    kept_rows += 1

    return {
        "shards": len(shard_paths),
        "input_rows": input_rows,
        "kept_lines": kept_rows,
    }


def main() -> None:
    args = parse_args()
    split_to_output = {
        "train": args.output_dir / "train.txt",
        "validation": args.output_dir / "valid.txt",
        "test": args.output_dir / "test.txt",
    }

    for split_name, output_path in split_to_output.items():
        stats = convert_split(
            split_name=split_name,
            source_dir=args.source_dir,
            output_path=output_path,
            keep_empty_lines=args.keep_empty_lines,
            strip_lines=args.strip_lines,
        )
        print(
            f"{split_name}: shards={stats['shards']} rows={stats['input_rows']} "
            f"kept_lines={stats['kept_lines']} -> {output_path}"
        )


if __name__ == "__main__":
    main()

