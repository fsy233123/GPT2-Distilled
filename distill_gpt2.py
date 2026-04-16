from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, GPT2Config, GPT2LMHeadModel

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
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "models" / "results.csv"
FORMAL_BASELINE_DIR = Path("/data/qyq/GPT2_compression")
FORMAL_SCRATCH_DIR = Path("/data/qyq/GPT2_compression_student_scratch")
FORMAL_DISTILLED_DIR = Path("/data/qyq/GPT2_compression_student_distilled")
LEGACY_BASELINE_DIR = PROJECT_ROOT / "models" / "baseline_gpt2"
LEGACY_SCRATCH_DIR = PROJECT_ROOT / "models" / "student_scratch"
LEGACY_DISTILLED_DIR = PROJECT_ROOT / "models" / "student_distilled"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a smaller GPT-2 student model.")
    parser.add_argument("--mode", choices=["scratch", "distill"], default="distill")
    parser.add_argument("--teacher_model_name_or_path", type=str, default=None)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--valid_file", type=Path, default=DEFAULT_VALID_FILE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--results_path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--student_layers", type=int, default=6)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--student_init",
        choices=["auto", "random", "teacher_copy"],
        default="auto",
        help="Initialization strategy for the student. "
        "'auto' uses teacher_copy in distill mode and random in scratch mode.",
    )
    parser.add_argument("--max_train_blocks", type=int, default=None)
    parser.add_argument("--max_valid_blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def resolve_teacher_path(args: argparse.Namespace) -> str:
    if args.teacher_model_name_or_path:
        return args.teacher_model_name_or_path
    if FORMAL_BASELINE_DIR.exists():
        return str(FORMAL_BASELINE_DIR)
    if LEGACY_BASELINE_DIR.exists():
        return str(LEGACY_BASELINE_DIR)
    return "gpt2"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.mode == "scratch":
        return FORMAL_SCRATCH_DIR
    return FORMAL_DISTILLED_DIR


def build_student_config(teacher_source: str, args: argparse.Namespace) -> GPT2Config:
    teacher_config = AutoConfig.from_pretrained(
        teacher_source,
        local_files_only=args.local_files_only,
    )
    return GPT2Config(
        vocab_size=teacher_config.vocab_size,
        n_positions=teacher_config.n_positions,
        n_ctx=teacher_config.n_ctx,
        n_embd=teacher_config.n_embd,
        n_layer=args.student_layers,
        n_head=teacher_config.n_head,
        n_inner=teacher_config.n_inner,
        activation_function=teacher_config.activation_function,
        resid_pdrop=teacher_config.resid_pdrop,
        embd_pdrop=teacher_config.embd_pdrop,
        attn_pdrop=teacher_config.attn_pdrop,
        layer_norm_epsilon=teacher_config.layer_norm_epsilon,
        initializer_range=teacher_config.initializer_range,
        bos_token_id=teacher_config.bos_token_id,
        eos_token_id=teacher_config.eos_token_id,
        pad_token_id=teacher_config.pad_token_id,
        use_cache=False,
    )


def resolve_student_init_strategy(args: argparse.Namespace) -> str:
    if args.student_init != "auto":
        return args.student_init
    return "teacher_copy" if args.mode == "distill" else "random"


def select_teacher_layer_indices(
    num_teacher_layers: int,
    num_student_layers: int,
) -> list[int]:
    if num_student_layers > num_teacher_layers:
        raise ValueError(
            "Student layers cannot exceed teacher layers for teacher-copy initialization: "
            f"{num_student_layers} > {num_teacher_layers}"
        )
    if num_student_layers == 1:
        return [num_teacher_layers - 1]

    positions = torch.linspace(0, num_teacher_layers - 1, steps=num_student_layers)
    indices = [int(round(position.item())) for position in positions]
    for idx in range(1, len(indices)):
        indices[idx] = max(indices[idx], indices[idx - 1] + 1)
    return [min(index, num_teacher_layers - 1) for index in indices]


def initialize_student_from_teacher(
    student_model: GPT2LMHeadModel,
    teacher_model: GPT2LMHeadModel | None,
    strategy: str,
) -> list[int]:
    if strategy == "random" or teacher_model is None:
        return []

    teacher_layer_indices = select_teacher_layer_indices(
        num_teacher_layers=len(teacher_model.transformer.h),
        num_student_layers=len(student_model.transformer.h),
    )

    with torch.no_grad():
        student_model.transformer.wte.weight.copy_(teacher_model.transformer.wte.weight)
        student_model.transformer.wpe.weight.copy_(teacher_model.transformer.wpe.weight)
        student_model.transformer.ln_f.load_state_dict(teacher_model.transformer.ln_f.state_dict())

        for student_layer_idx, teacher_layer_idx in enumerate(teacher_layer_indices):
            student_model.transformer.h[student_layer_idx].load_state_dict(
                teacher_model.transformer.h[teacher_layer_idx].state_dict()
            )

        if hasattr(student_model, "lm_head") and hasattr(teacher_model, "lm_head"):
            student_model.lm_head.weight.copy_(teacher_model.lm_head.weight)

    student_model.tie_weights()
    return teacher_layer_indices


def main() -> None:
    args = parse_args()
    distributed_context = setup_distributed()
    try:
        set_seed(args.seed)
        device = distributed_context.device if distributed_context.is_distributed else get_device()
        is_main = distributed_context.is_main_process

        teacher_source = resolve_teacher_path(args)
        output_dir = resolve_output_dir(args)
        student_init_strategy = resolve_student_init_strategy(args)

        if is_main:
            print("[Stage 1/7] Loading tokenizer")
        tokenizer = load_tokenizer(
            teacher_source,
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

        teacher_model = None
        if args.mode == "distill":
            if is_main:
                print("[Stage 4/7] Loading teacher model")
            teacher_model = load_pretrained_model(
                GPT2LMHeadModel,
                teacher_source,
                local_files_only=args.local_files_only,
            )
            teacher_model.config.pad_token_id = tokenizer.pad_token_id
            teacher_model.requires_grad_(False)
            teacher_model.to(device)
            teacher_model.eval()

        if is_main:
            print("[Stage 5/7] Building student model")
        student_config = build_student_config(teacher_source, args)
        student_model = GPT2LMHeadModel(student_config)
        student_model.config.pad_token_id = tokenizer.pad_token_id
        student_model.to(device)
        copied_layers = initialize_student_from_teacher(
            student_model=student_model,
            teacher_model=teacher_model,
            strategy=student_init_strategy,
        )
        if is_main:
            if copied_layers:
                print(
                    "[Init] Student initialized from teacher layers: "
                    + ", ".join(str(index) for index in copied_layers)
                )
            else:
                print("[Init] Student initialized randomly")
        student_model = wrap_model_for_distributed(student_model, distributed_context)

        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        if is_main:
            print("[Stage 6/7] Training and validation")
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            stats = train_epoch(
                model=student_model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_accum_steps=args.grad_accum_steps,
                teacher_model=teacher_model,
                alpha=args.alpha,
                temperature=args.temperature,
                use_amp=args.use_amp,
                distributed_context=distributed_context,
            )
            valid_loss, valid_ppl = evaluate_perplexity(
                model=student_model,
                dataloader=valid_loader,
                device=device,
                use_amp=args.use_amp,
                distributed_context=distributed_context,
            )
            if distributed_context.is_main_process:
                print(
                    f"[Epoch {epoch}] train_loss={stats.train_loss:.4f} "
                    f"ce_loss={stats.ce_loss:.4f} kl_loss={stats.kl_loss:.4f} "
                    f"valid_loss={valid_loss:.4f} valid_ppl={valid_ppl:.4f}"
                )

        barrier()
        base_model = unwrap_model(student_model)
        if distributed_context.is_main_process:
            print("[Stage 7/7] Saving outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            base_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            row_name = "Student scratch" if args.mode == "scratch" else "Student distilled"
            result_row = build_result_row(
                model_name=row_name,
                params=count_parameters(base_model),
                sparsity=0.0,
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
