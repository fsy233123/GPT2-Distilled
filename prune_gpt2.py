from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

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
    count_nonzero_parameters,
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
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "models" / "results.csv"
FORMAL_DISTILLED_DIR = Path("/data/qyq/GPT2_compression_student_distilled")
FORMAL_PRUNED_PREFIX = "/data/qyq/GPT2_compression_student_pruned"
LEGACY_STUDENT_DIR = PROJECT_ROOT / "models" / "student_distilled"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune and recover a distilled GPT-2 student.")
    parser.add_argument("--model_path", type=Path, default=FORMAL_DISTILLED_DIR)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--valid_file", type=Path, default=DEFAULT_VALID_FILE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--results_path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sparsity", type=float, default=0.4, choices=[0.2, 0.4, 0.6])
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--finetune_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_train_blocks", type=int, default=None)
    parser.add_argument("--max_valid_blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def collect_prunable_modules(model: GPT2LMHeadModel):
    modules = []
    for block in model.transformer.h:
        for module_name in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            module = block.get_submodule(module_name)
            if isinstance(module, (torch.nn.Linear, Conv1D)):
                modules.append(module)
    return modules


def compute_nonzero_ratio(modules) -> tuple[int, int, float]:
    nonzero, total = count_nonzero_parameters(module.weight for module in modules)
    ratio = nonzero / total if total else 0.0
    return nonzero, total, ratio


def solidify_pruning(modules) -> dict[int, torch.Tensor]:
    masks = {}
    for module in modules:
        prune.remove(module, "weight")
        masks[id(module)] = module.weight.detach().ne(0).to(module.weight.device)
    return masks


def enforce_pruning_masks(modules, masks: dict[int, torch.Tensor]) -> None:
    with torch.no_grad():
        for module in modules:
            module.weight.mul_(masks[id(module)])


def resolve_student_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    if model_path == FORMAL_DISTILLED_DIR and LEGACY_STUDENT_DIR.exists():
        return LEGACY_STUDENT_DIR
    return model_path


def main() -> None:
    args = parse_args()
    distributed_context = setup_distributed()
    try:
        set_seed(args.seed)
        device = distributed_context.device if distributed_context.is_distributed else get_device()
        is_main = distributed_context.is_main_process

        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else Path(f"{FORMAL_PRUNED_PREFIX}_{int(args.sparsity * 100)}")
        )
        model_path = resolve_student_model_path(args.model_path)

        if is_main:
            print("[Stage 1/7] Loading tokenizer")
        tokenizer = load_tokenizer(
            str(model_path),
            local_files_only=args.local_files_only,
        )
        if is_main:
            print("[Stage 2/7] Building training dataloader")
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
            print("[Stage 3/7] Building validation dataloader")
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
            print("[Stage 4/7] Loading distilled student")
        model = load_pretrained_model(
            GPT2LMHeadModel,
            str(model_path),
            local_files_only=args.local_files_only,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)

        prunable_modules = collect_prunable_modules(model)
        if not prunable_modules:
            raise RuntimeError("No prunable attention/MLP projection modules were found.")

        if is_main:
            print("[Stage 5/7] Applying pruning and evaluating pre-recovery PPL")
        before_nonzero, total_prunable, before_ratio = compute_nonzero_ratio(prunable_modules)
        prune.global_unstructured(
            [(module, "weight") for module in prunable_modules],
            pruning_method=prune.L1Unstructured,
            amount=args.sparsity,
        )
        after_nonzero, _, after_ratio = compute_nonzero_ratio(prunable_modules)

        if distributed_context.is_main_process:
            print(
                f"Prunable weights before pruning: {before_nonzero}/{total_prunable} "
                f"non-zero ({before_ratio * 100:.2f}%)"
            )
            print(
                f"Prunable weights after pruning : {after_nonzero}/{total_prunable} "
                f"non-zero ({after_ratio * 100:.2f}%)"
            )

        valid_loss_before_ft, valid_ppl_before_ft = evaluate_perplexity(
            model=model,
            dataloader=valid_loader,
            device=device,
            use_amp=args.use_amp,
            distributed_context=distributed_context,
        )
        if distributed_context.is_main_process:
            print(
                f"Validation before recovery fine-tune: "
                f"loss={valid_loss_before_ft:.4f} ppl={valid_ppl_before_ft:.4f}"
            )

        pruning_masks = solidify_pruning(prunable_modules)
        model = wrap_model_for_distributed(model, distributed_context)

        valid_loss = valid_loss_before_ft
        valid_ppl = valid_ppl_before_ft
        if args.finetune_epochs > 0:
            if is_main:
                print("[Stage 6/7] Recovery fine-tuning")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            for epoch in range(1, args.finetune_epochs + 1):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                stats = train_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    grad_accum_steps=args.grad_accum_steps,
                    use_amp=args.use_amp,
                    post_step_callback=lambda: enforce_pruning_masks(prunable_modules, pruning_masks),
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
                        f"[Epoch {epoch}] recovery_train_loss={stats.train_loss:.4f} "
                        f"valid_loss={valid_loss:.4f} valid_ppl={valid_ppl:.4f}"
                    )

        base_model = unwrap_model(model)
        final_nonzero, _, final_ratio = compute_nonzero_ratio(collect_prunable_modules(base_model))
        realized_sparsity = 1.0 - final_ratio
        if distributed_context.is_main_process:
            print(
                f"Final prunable weights after recovery: {final_nonzero}/{total_prunable} "
                f"non-zero ({final_ratio * 100:.2f}%)"
            )

        barrier()
        if distributed_context.is_main_process:
            print("[Stage 7/7] Saving outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            base_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            result_row = build_result_row(
                model_name="Distilled + Pruned",
                params=count_parameters(base_model),
                sparsity=realized_sparsity,
                ppl=valid_ppl,
                model_path=output_dir,
            )
            append_result_row(args.results_path, result_row)
            print("\nCurrent comparison table:")
            print(format_markdown_table(load_result_rows(args.results_path)))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
