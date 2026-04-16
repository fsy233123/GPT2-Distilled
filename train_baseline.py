from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from utils.data_utils import create_dataloader, load_tokenizer
from utils.distributed import (
    barrier,
    cleanup_distributed,
    setup_distributed,
    unwrap_model,
    wrap_model_for_distributed,
)
from utils.hf_utils import load_pretrained_model
from utils.metrics import (
    append_result_row,
    build_result_row,
    format_markdown_table,
    load_result_rows,
)
from utils.training import (
    count_parameters,
    evaluate_perplexity,
    get_device,
    set_seed,
    train_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data" / "train.txt"
DEFAULT_VALID_FILE = PROJECT_ROOT / "data" / "valid.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "baseline_gpt2"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "models" / "results.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a GPT-2 baseline model.")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--valid_file", type=Path, default=DEFAULT_VALID_FILE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results_path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_train_blocks", type=int, default=None)
    parser.add_argument("--max_valid_blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed_context = setup_distributed()
    try:
        set_seed(args.seed)
        device = distributed_context.device if distributed_context.is_distributed else get_device()
        is_main = distributed_context.is_main_process

        if is_main:
            print("[Stage 1/6] Loading tokenizer")

        tokenizer = load_tokenizer(
            args.model_name_or_path,
            local_files_only=args.local_files_only,
        )
        if is_main:
            print("[Stage 2/6] Building training dataloader")
        train_loader, train_sampler = create_dataloader(
            text_path=args.train_file,
            tokenizer=tokenizer,
            block_size=args.block_size,
            batch_size=args.batch_size,
            shuffle=True,
            max_blocks=args.max_train_blocks,
            num_workers=args.num_workers,
            distributed=distributed_context.is_distributed,
            rank=distributed_context.rank,
            world_size=distributed_context.world_size,
            cache_dir=args.cache_dir,
            show_progress=is_main,
        )
        if is_main:
            print("[Stage 3/6] Building validation dataloader")
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
            print("[Stage 4/6] Loading baseline model")
        model = load_pretrained_model(
            GPT2LMHeadModel,
            args.model_name_or_path,
            local_files_only=args.local_files_only,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        model = wrap_model_for_distributed(model, distributed_context)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        if is_main:
            print("[Stage 5/6] Training and validation")
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            stats = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_accum_steps=args.grad_accum_steps,
                use_amp=args.use_amp,
                distributed_context=distributed_context,
            )
            valid_loss, valid_ppl = evaluate_perplexity(
                model=model,
                dataloader=valid_loader,
                device=device,
                use_amp=args.use_amp,
                distributed_context=distributed_context,
            )
            if distributed_context.is_main_process:
                print(
                    f"[Epoch {epoch}] train_loss={stats.train_loss:.4f} "
                    f"valid_loss={valid_loss:.4f} valid_ppl={valid_ppl:.4f}"
                )

        barrier()
        base_model = unwrap_model(model)
        if distributed_context.is_main_process:
            print("[Stage 6/6] Saving outputs")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            base_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            params = count_parameters(base_model)
            result_row = build_result_row(
                model_name="GPT2 baseline",
                params=params,
                sparsity=0.0,
                ppl=valid_ppl,
                model_path=args.output_dir,
            )
            append_result_row(args.results_path, result_row)
            print("\nCurrent comparison table:")
            print(format_markdown_table(load_result_rows(args.results_path)))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
