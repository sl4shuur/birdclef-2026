"""Portable, incremental embedding cache; targets and folds stay in prepared data."""

import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.progress import track
from torch.utils.data import Dataset, Subset

from birdclef.config import LoaderConfig
from birdclef.data.datamodule import prediction_dataloader
from birdclef.models.classifier.perch.encoder import PerchEncoder


def _cache_path(row: dict, directory: Path) -> Path:
    # Audio is immutable competition data. Size detects replacements without hashing GBs
    # every epoch; names and offsets preserve identity across machines and dataset mounts.
    focal = row.get("source") == "focal"
    identity = [
        "focal" if focal else "soundscape",
        row["filename"],
        "center" if focal else float(row["start"]),
        Path(row["path"]).stat().st_size,
    ]
    key = sha256(json.dumps(identity).encode()).hexdigest()
    return directory / f"{key}.npy"


def _save(path: Path, embedding: torch.Tensor) -> None:
    # Unique temporary files also make concurrent extraction safe.
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
    encoder: PerchEncoder,
    *,
    loader: LoaderConfig,
    cache_dir: Path | None = None,
) -> torch.Tensor:
    """Read existing entries, extract only misses and return the original sample order.

    A complete cache works on a read-only Kaggle Dataset. With cache_dir=None,
    inference extracts directly without trying to write into a training cache.
    Bump the namespace if decoding, resampling or crop policy changes.
    """
    paths = []
    if cache_dir is not None:
        directory = cache_dir / "perch-center-v1" / encoder.sha256
        paths = [_cache_path(row, directory) for row in frame.to_dict("records")]
    missing = [index for index in range(len(frame)) if not paths or not paths[index].is_file()]
    if missing and paths:
        paths[0].parent.mkdir(parents=True, exist_ok=True)
    values: dict[int, torch.Tensor] = {}
    batches = prediction_dataloader(Subset(dataset, missing), config=loader)
    offset = 0
    for batch in track(batches, description=f"Perch: {len(missing)} uncached windows"):
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        for embedding in encoder(inputs):
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
