from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch


RESULT_HEADERS = ["model", "params", "sparsity", "ppl", "model_path"]


def _normalize_row(row: dict) -> dict[str, str]:
    return {key: row.get(key, "") for key in RESULT_HEADERS}


def _load_result_rows_raw(results_path: str | Path) -> list[dict[str, str]]:
    results_path = Path(results_path)
    if not results_path.exists():
        return []

    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_normalize_row(row) for row in reader]


def _deduplicate_result_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduplicated: dict[str, dict[str, str]] = {}
    for row in rows:
        model_name = row.get("model", "")
        deduplicated[model_name] = _normalize_row(row)
    return list(deduplicated.values())


def _write_result_rows(results_path: str | Path, rows: list[dict[str, str]]) -> None:
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))


def append_result_row(results_path: str | Path, row: dict) -> None:
    rows = _deduplicate_result_rows(_load_result_rows_raw(results_path))
    normalized_row = _normalize_row(row)
    row_by_model = {existing_row["model"]: existing_row for existing_row in rows}
    row_by_model[normalized_row["model"]] = normalized_row
    _write_result_rows(results_path, list(row_by_model.values()))


def load_result_rows(results_path: str | Path) -> list[dict[str, str]]:
    return _deduplicate_result_rows(_load_result_rows_raw(results_path))


def build_result_row(
    model_name: str,
    params: int,
    sparsity: float,
    ppl: float,
    model_path: str | Path,
) -> dict[str, str]:
    return {
        "model": model_name,
        "params": str(params),
        "sparsity": f"{sparsity * 100:.1f}%",
        "ppl": f"{ppl:.4f}",
        "model_path": str(model_path),
    }


def format_markdown_table(rows: Iterable[dict[str, str]]) -> str:
    rows = list(rows)
    if not rows:
        return "No results recorded yet."

    header = "| Model | Params | Sparsity | PPL |"
    divider = "| --- | --- | --- | --- |"
    body = [
        f"| {row['model']} | {row['params']} | {row['sparsity']} | {row['ppl']} |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def count_nonzero_parameters(parameters: Iterable[torch.Tensor]) -> tuple[int, int]:
    nonzero = 0
    total = 0
    for parameter in parameters:
        tensor = parameter.detach()
        nonzero += torch.count_nonzero(tensor).item()
        total += tensor.numel()
    return nonzero, total
