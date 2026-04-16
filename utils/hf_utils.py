from __future__ import annotations

from pathlib import Path


HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _repo_id_to_cache_dir(repo_id: str) -> Path:
    return HF_CACHE_ROOT / f"models--{repo_id.replace('/', '--')}"


def find_local_snapshot_with_weights(model_name_or_path: str) -> str | None:
    model_path = Path(model_name_or_path)
    if model_path.exists():
        if any((model_path / filename).exists() for filename in MODEL_WEIGHT_FILES):
            return str(model_path)
        return None

    snapshot_root = _repo_id_to_cache_dir(model_name_or_path) / "snapshots"
    if not snapshot_root.exists():
        return None

    candidates = []
    for snapshot_dir in snapshot_root.iterdir():
        if snapshot_dir.is_dir() and any(
            (snapshot_dir / filename).exists() for filename in MODEL_WEIGHT_FILES
        ):
            candidates.append(snapshot_dir)

    if not candidates:
        return None
    return str(sorted(candidates)[-1])


def load_pretrained_model(model_cls, model_name_or_path: str, local_files_only: bool = False, **kwargs):
    try:
        return model_cls.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            **kwargs,
        )
    except AttributeError as exc:
        if not local_files_only:
            raise

        snapshot_path = find_local_snapshot_with_weights(model_name_or_path)
        if snapshot_path is None:
            raise RuntimeError(
                "No complete local checkpoint weights were found for "
                f"'{model_name_or_path}'. Download the model once without "
                "--local_files_only, or point the argument to a local model "
                "directory that already contains model weights."
            ) from exc

        return model_cls.from_pretrained(
            snapshot_path,
            local_files_only=True,
            **kwargs,
        )

