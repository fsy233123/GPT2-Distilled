from __future__ import annotations

import argparse
from pathlib import Path

from utils.data_utils import TextBlockDataset, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data" / "train.txt"
DEFAULT_VALID_FILE = PROJECT_ROOT / "data" / "valid.txt"
DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "test.txt"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize text data and save GPT-2 block caches to disk."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data/lmy_data/LLM/GPT2",
        help="Tokenizer path used during preprocessing.",
    )
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--valid_file", type=Path, default=DEFAULT_VALID_FILE)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_train_blocks", type=int, default=None)
    parser.add_argument("--max_valid_blocks", type=int, default=None)
    parser.add_argument("--max_test_blocks", type=int, default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_valid", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def build_cache(
    split_name: str,
    text_path: Path,
    tokenizer,
    block_size: int,
    cache_dir: Path,
    max_blocks: int | None,
) -> None:
    print(f"[Cache] Preparing {split_name} from {text_path}")
    dataset = TextBlockDataset(
        text_path=text_path,
        tokenizer=tokenizer,
        block_size=block_size,
        max_blocks=max_blocks,
        cache_dir=cache_dir,
        rank=0,
        show_progress=True,
    )
    print(
        f"[Cache] Ready: split={split_name} blocks={len(dataset)} "
        f"block_size={block_size}"
    )


def main() -> None:
    args = parse_args()

    print("[Step 1/2] Loading tokenizer")
    tokenizer = load_tokenizer(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )

    print("[Step 2/2] Building dataset caches")
    if not args.skip_train:
        build_cache(
            split_name="train",
            text_path=args.train_file,
            tokenizer=tokenizer,
            block_size=args.block_size,
            cache_dir=args.cache_dir,
            max_blocks=args.max_train_blocks,
        )
    if not args.skip_valid:
        build_cache(
            split_name="valid",
            text_path=args.valid_file,
            tokenizer=tokenizer,
            block_size=args.block_size,
            cache_dir=args.cache_dir,
            max_blocks=args.max_valid_blocks,
        )
    if not args.skip_test and args.test_file.exists():
        build_cache(
            split_name="test",
            text_path=args.test_file,
            tokenizer=tokenizer,
            block_size=args.block_size,
            cache_dir=args.cache_dir,
            max_blocks=args.max_test_blocks,
        )

    print(f"[Done] Cache directory: {args.cache_dir}")


if __name__ == "__main__":
    main()

