from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.distributed import DistributedContext, all_reduce_tensor


@dataclass
class EpochStats:
    train_loss: float
    ce_loss: float
    kl_loss: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def get_autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def shift_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels


def causal_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_shift, shift_labels = shift_logits_and_labels(student_logits, labels)
    teacher_shift, _ = shift_logits_and_labels(teacher_logits, labels)
    valid_mask = shift_labels.ne(-100).view(-1)

    student_flat = student_shift.view(-1, student_shift.size(-1))[valid_mask]
    teacher_flat = teacher_shift.view(-1, teacher_shift.size(-1))[valid_mask]

    loss = F.kl_div(
        F.log_softmax(student_flat / temperature, dim=-1),
        F.softmax(teacher_flat / temperature, dim=-1),
        reduction="batchmean",
    )
    return loss * (temperature ** 2)


def evaluate_perplexity(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    use_amp: bool = False,
    distributed_context: Optional[DistributedContext] = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        iterator = dataloader
        if distributed_context is None or distributed_context.is_main_process:
            iterator = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in iterator:
            batch = move_batch_to_device(batch, device)
            with get_autocast_context(device, use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                loss = causal_ce_loss(outputs.logits, batch["labels"])

            valid_tokens = batch["labels"][..., 1:].ne(-100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    if distributed_context is not None and distributed_context.is_distributed:
        aggregate = torch.tensor(
            [total_loss, float(total_tokens)],
            dtype=torch.float64,
            device=device,
        )
        all_reduce_tensor(aggregate)
        total_loss = aggregate[0].item()
        total_tokens = int(aggregate[1].item())

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 20))
    return mean_loss, perplexity


def train_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    teacher_model: Optional[torch.nn.Module] = None,
    alpha: float = 0.5,
    temperature: float = 2.0,
    use_amp: bool = False,
    max_grad_norm: Optional[float] = 1.0,
    post_step_callback: Optional[Callable[[], None]] = None,
    distributed_context: Optional[DistributedContext] = None,
) -> EpochStats:
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=use_amp and device.type == "cuda",
        )
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_tokens = 0

    progress = dataloader
    if distributed_context is None or distributed_context.is_main_process:
        progress = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        valid_tokens = batch["labels"][..., 1:].ne(-100).sum().item()
        should_step = step % grad_accum_steps == 0 or step == len(dataloader)
        sync_context = nullcontext()
        if (
            distributed_context is not None
            and distributed_context.is_distributed
            and hasattr(model, "no_sync")
            and not should_step
        ):
            sync_context = model.no_sync()

        with sync_context:
            with get_autocast_context(device, use_amp):
                student_outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                ce_loss = causal_ce_loss(student_outputs.logits, batch["labels"])

                kl_loss = torch.tensor(0.0, device=device)
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=False,
                        )
                    kl_loss = distillation_kl_loss(
                        student_logits=student_outputs.logits,
                        teacher_logits=teacher_outputs.logits,
                        labels=batch["labels"],
                        temperature=temperature,
                    )
                    loss = alpha * ce_loss + (1.0 - alpha) * kl_loss
                else:
                    loss = ce_loss

        loss_for_backward = loss / grad_accum_steps
        if scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if should_step:
            if max_grad_norm is not None:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if post_step_callback is not None:
                post_step_callback()

        total_loss += loss.item() * valid_tokens
        total_ce += ce_loss.item() * valid_tokens
        total_kl += kl_loss.item() * valid_tokens
        total_tokens += valid_tokens

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{ce_loss.item():.4f}",
                kl=f"{kl_loss.item():.4f}",
            )

    if distributed_context is not None and distributed_context.is_distributed:
        aggregate = torch.tensor(
            [total_loss, total_ce, total_kl, float(total_tokens)],
            dtype=torch.float64,
            device=device,
        )
        all_reduce_tensor(aggregate)
        total_loss = aggregate[0].item()
        total_ce = aggregate[1].item()
        total_kl = aggregate[2].item()
        total_tokens = int(aggregate[3].item())

    return EpochStats(
        train_loss=total_loss / max(total_tokens, 1),
        ce_loss=total_ce / max(total_tokens, 1),
        kl_loss=total_kl / max(total_tokens, 1),
    )
