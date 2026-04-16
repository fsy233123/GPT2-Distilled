from __future__ import annotations

import argparse
from pathlib import Path

from transformers import GPT2LMHeadModel

from utils.data_utils import create_dataloader, load_tokenizer
from utils.distributed import cleanup_distributed, setup_distributed, unwrap_model, wrap_model_for_distributed
from utils.hf_utils import load_pretrained_model
from utils.training import count_parameters, evaluate_perplexity, get_device


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VALID_FILE = PROJECT_ROOT / "data" / "valid.txt"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation perplexity.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--valid_file", type=Path, default=DEFAULT_VALID_FILE)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--max_valid_blocks", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed_context = setup_distributed()
    try:
        device = distributed_context.device if distributed_context.is_distributed else get_device()
        is_main = distributed_context.is_main_process

        if is_main:
            print("[Stage 1/3] Loading tokenizer and validation data")
        tokenizer = load_tokenizer(
            args.model_path,
            local_files_only=args.local_files_only,
        )
        valid_loader, _ = create_dataloader(
            text_path=args.valid_file,
            tokenizer=tokenizer,
            block_size=args.block_size,
            batch_size=args.eval_batch_size,
            shuffle=False,
            max_blocks=args.max_valid_blocks,
            num_workers=args.num_workers,
            distributed=distributed_context.is_distributed,
            rank=distributed_context.rank,
            world_size=distributed_context.world_size,
            cache_dir=args.cache_dir,
            show_progress=is_main,
        )

        if is_main:
            print("[Stage 2/3] Loading model")
        model = load_pretrained_model(
            GPT2LMHeadModel,
            args.model_path,
            local_files_only=args.local_files_only,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        model = wrap_model_for_distributed(model, distributed_context)

        if is_main:
            print("[Stage 3/3] Evaluating perplexity")
        valid_loss, valid_ppl = evaluate_perplexity(
            model=model,
            dataloader=valid_loader,
            device=device,
            use_amp=args.use_amp,
            distributed_context=distributed_context,
        )
        if distributed_context.is_main_process:
            base_model = unwrap_model(model)
            print(f"Model: {args.model_path}")
            print(f"Params: {count_parameters(base_model)}")
            print(f"Validation loss: {valid_loss:.4f}")
            print(f"Validation perplexity: {valid_ppl:.4f}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
