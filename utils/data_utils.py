from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class TextBlockDataset(Dataset):
    """Builds fixed-length token blocks for causal language modeling."""

    def __init__(
        self,
        text_path: str | Path,
        tokenizer,
        block_size: int,
        max_blocks: Optional[int] = None,
        cache_dir: str | Path | None = None,
        rank: int = 0,
        show_progress: bool = True,
    ) -> None:
        self.text_path = Path(text_path)
        if not self.text_path.exists():
            raise FileNotFoundError(f"Text file not found: {self.text_path}")
        self.examples = self._load_or_build_examples(
            tokenizer=tokenizer,
            block_size=block_size,
            max_blocks=max_blocks,
            cache_dir=cache_dir,
            rank=rank,
            show_progress=show_progress,
        )

    def _load_or_build_examples(
        self,
        tokenizer,
        block_size: int,
        max_blocks: Optional[int],
        cache_dir: str | Path | None,
        rank: int,
        show_progress: bool,
    ) -> list[list[int]]:
        cache_path = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = self._build_cache_key(tokenizer, block_size, max_blocks)
            cache_path = cache_dir / f"{self.text_path.stem}_{cache_key}.pt"

        if cache_path is not None and cache_path.exists():
            if show_progress:
                print(f"[Data] Loading cached blocks from {cache_path}")
            return self._load_cache_file(cache_path)

        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed and rank != 0 and cache_path is not None:
            return self._wait_and_load_cache_file(cache_path)

        examples = self._build_examples_from_text(
            tokenizer=tokenizer,
            block_size=block_size,
            max_blocks=max_blocks,
            show_progress=show_progress,
        )

        if cache_path is not None:
            if show_progress:
                print(f"[Data] Saving cached blocks to {cache_path}")
            self._atomic_save_cache_file(examples, cache_path)

        return examples

    def _load_cache_file(self, cache_path: Path) -> list[list[int]]:
        try:
            return torch.load(cache_path, map_location="cpu")
        except (EOFError, RuntimeError, pickle.UnpicklingError):
            if cache_path.exists():
                cache_path.unlink()
            raise

    def _wait_and_load_cache_file(
        self,
        cache_path: Path,
        poll_interval_seconds: float = 5.0,
        log_interval_seconds: float = 60.0,
    ) -> list[list[int]]:
        start_time = time.time()
        last_log_time = 0.0
        while True:
            if cache_path.exists():
                try:
                    return torch.load(cache_path, map_location="cpu")
                except (EOFError, RuntimeError, pickle.UnpicklingError):
                    pass

            elapsed = time.time() - start_time
            if elapsed - last_log_time >= log_interval_seconds:
                print(
                    f"[Data] Waiting for cache from rank 0: {cache_path.name} "
                    f"(elapsed {elapsed / 60:.1f} min)"
                )
                last_log_time = elapsed
            time.sleep(poll_interval_seconds)

    def _atomic_save_cache_file(self, examples: list[list[int]], cache_path: Path) -> None:
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        if temp_path.exists():
            temp_path.unlink()
        torch.save(examples, temp_path)
        temp_path.replace(cache_path)

    def _wait_for_cache_file(
        self,
        cache_path: Path,
        poll_interval_seconds: float = 5.0,
        log_interval_seconds: float = 60.0,
    ) -> None:
        start_time = time.time()
        last_log_time = 0.0
        while not cache_path.exists():
            elapsed = time.time() - start_time
            if elapsed - last_log_time >= log_interval_seconds:
                print(
                    f"[Data] Waiting for cache from rank 0: {cache_path.name} "
                    f"(elapsed {elapsed / 60:.1f} min)"
                )
                last_log_time = elapsed
            time.sleep(poll_interval_seconds)

    def _build_cache_key(
        self,
        tokenizer,
        block_size: int,
        max_blocks: Optional[int],
    ) -> str:
        tokenizer_name = getattr(tokenizer, "name_or_path", "tokenizer")
        payload = (
            f"{self.text_path.resolve()}|{tokenizer_name}|"
            f"{tokenizer.eos_token_id}|{block_size}|{max_blocks}"
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]

    def _build_examples_from_text(
        self,
        tokenizer,
        block_size: int,
        max_blocks: Optional[int],
        show_progress: bool,
    ) -> list[list[int]]:
        file_size = self.text_path.stat().st_size
        progress = tqdm(
            total=file_size,
            desc=f"Tokenizing {self.text_path.name}",
            unit="B",
            unit_scale=True,
            leave=False,
            disable=not show_progress,
        )

        examples: list[list[int]] = []
        token_buffer: list[int] = []
        seen_any_text = False

        with self.text_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                progress.update(len(raw_line.encode("utf-8")))
                if not raw_line:
                    continue
                seen_any_text = True
                token_buffer.extend(
                    tokenizer.encode(raw_line, add_special_tokens=False)
                )

                while len(token_buffer) >= block_size:
                    examples.append(token_buffer[:block_size])
                    token_buffer = token_buffer[block_size:]
                    if max_blocks is not None and len(examples) >= max_blocks:
                        progress.close()
                        return examples

        if tokenizer.eos_token_id is not None:
            token_buffer.append(tokenizer.eos_token_id)
            while len(token_buffer) >= block_size:
                examples.append(token_buffer[:block_size])
                token_buffer = token_buffer[block_size:]
                if max_blocks is not None and len(examples) >= max_blocks:
                    progress.close()
                    return examples

        progress.close()

        if not seen_any_text:
            raise ValueError(f"Text file is empty: {self.text_path}")

        if not examples:
            if not token_buffer:
                raise ValueError(f"Text file is empty after tokenization: {self.text_path}")
            repeats = (block_size // max(len(token_buffer), 1)) + 2
            expanded = (token_buffer * repeats)[:block_size]
            examples = [expanded]

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.examples[index], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def load_tokenizer(model_name_or_path: str, local_files_only: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    tokenizer.model_max_length = int(1e30)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_dataloader(
    text_path: str | Path,
    tokenizer,
    block_size: int,
    batch_size: int,
    shuffle: bool,
    max_blocks: Optional[int] = None,
    num_workers: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    cache_dir: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = TextBlockDataset(
        text_path=text_path,
        tokenizer=tokenizer,
        block_size=block_size,
        max_blocks=max_blocks,
        cache_dir=cache_dir,
        rank=rank,
        show_progress=show_progress,
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataloader, sampler
