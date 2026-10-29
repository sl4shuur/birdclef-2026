"""Portable, incremental embedding cache; targets and folds stay in prepared data."""

import json
import os
import tempfile
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.progress import track
from torch.utils.data import Dataset, Subset

from birdclef.config import LoaderConfig
from birdclef.data.datamodule import prediction_dataloader


def _sample_identity(row: dict) -> list:
    focal = row.get("source") == "focal"
    return [
        "focal" if focal else "soundscape",
        row["filename"],
        "center" if focal else float(row["start"]),
    ]


def _cache_path(row: dict, directory: Path) -> Path:
    # Preserve existing array names when indexing immutable competition audio.
    identity = [*_sample_identity(row), Path(row["path"]).stat().st_size]
    key = sha256(json.dumps(identity).encode()).hexdigest()
    return directory / f"{key}.npy"


def _cache_paths(frame: pd.DataFrame, directory: Path, *, cache_only: bool) -> list[Path]:
    """Resolve saved sample identities without reopening their source audio."""
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    paths = []
    changed = False
    for row in frame.to_dict("records"):
        key = json.dumps(_sample_identity(row))
        if key not in index:
            if cache_only:
                raise ValueError(
                    "Cache index is incomplete; prepare embeddings in one process before DDP."
                )
            # Existing .npy names stay intact; only new/unindexed samples need audio metadata.
            index[key] = _cache_path(row, directory).name
            changed = True
        paths.append(directory / index[key])
    if changed:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(index, stream)
        try:
            temporary.replace(index_path)
        finally:
            temporary.unlink(missing_ok=True)
    return paths


def _save(path: Path, embedding: torch.Tensor) -> None:
    # Atomic replacement prevents partial arrays; it does not coordinate extraction.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, embedding.numpy(), allow_pickle=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def embeddings(
    frame: pd.DataFrame,
    dataset: Dataset,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    *,
    loader: LoaderConfig,
    cache_dir: Path | None = None,
    cache_only: bool = False,
) -> torch.Tensor:
    """Read existing entries, extract only misses and return the original sample order.

    A complete cache works on a read-only Kaggle Dataset. With cache_dir=None,
    inference extracts directly without trying to write into a training cache.
    cache_dir is the complete namespace chosen by the caller. Bump it if decoding,
    resampling or crop policy changes. Targets and model construction stay outside.
    cache_only rejects misses without updating the index or running the extractor.
    """
    if len(frame) == 0:
        raise ValueError("Embedding extraction requires at least one sample.")
    paths = []
    if cache_dir is not None:
        paths = _cache_paths(frame, cache_dir, cache_only=cache_only)
    missing = [index for index in range(len(frame)) if not paths or not paths[index].is_file()]
    if missing and cache_only:
        raise ValueError("Cache entries are missing; prepare embeddings in one process before DDP.")
    if missing and paths:
        paths[0].parent.mkdir(parents=True, exist_ok=True)
    values: dict[int, torch.Tensor] = {}
    batches = prediction_dataloader(Subset(dataset, missing), config=loader)
    offset = 0
    for batch in track(batches, description=f"Embeddings: {len(missing)} uncached windows"):
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        features = encoder(inputs).detach().cpu()
        if (
            features.ndim != 2
            or features.shape[0] != len(inputs)
            or features.shape[1] == 0
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Encoder must return one finite feature vector per sample.")
        for embedding in features:
            index = missing[offset]
            if paths:
                _save(paths[index], embedding)
            else:
                values[index] = embedding
            offset += 1
    if paths:
        values = {
            index: torch.from_numpy(np.load(path, allow_pickle=False))
            for index, path in enumerate(paths)
        }
    result = torch.stack([values[index] for index in range(len(frame))])
    if result.ndim != 2 or not torch.isfinite(result).all():
        raise ValueError("Invalid embedding cache; remove the affected cache entries and retry.")
    return result
