from __future__ import annotations

import argparse
from pathlib import Path

from utils.metrics import format_markdown_table, load_result_rows


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "models" / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate results.csv so each model keeps only its latest row."
    )
    parser.add_argument("--results_path", type=Path, default=DEFAULT_RESULTS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_result_rows(args.results_path)
    if not rows:
        print("No results found.")
        return

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("model,params,sparsity,ppl,model_path\n")
        for row in rows:
            handle.write(
                f"{row['model']},{row['params']},{row['sparsity']},"
                f"{row['ppl']},{row['model_path']}\n"
            )

    print("Deduplicated results table:")
    print(format_markdown_table(rows))


if __name__ == "__main__":
    main()
