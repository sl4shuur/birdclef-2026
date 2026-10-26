"""Waveform datasets with explicit clip-level versus window-level supervision."""

from collections.abc import Callable
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from birdclef.competition import SAMPLE_RATE, SEGMENT_SECONDS, SOUNDSCAPE_SECONDS
from birdclef.data.features import feature_tensor
from birdclef.data.labels import align_targets

Transform = Callable[[torch.Tensor], torch.Tensor]


def read_audio_window(
    path: str | Path,
    *,
    start: float = 0,
    seconds: float = SEGMENT_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    pad: bool = False,
) -> torch.Tensor:
    """Seek at source rate, downmix, resample and optionally right-pad with zero.

    No amplitude normalization. Strict windows reject truncated soundscapes;
    padding must be explicitly enabled for short focal clips.
    """
    if not np.isfinite(start) or start < 0 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Audio start must be nonnegative and duration must be positive.")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    with sf.SoundFile(path) as audio:
        offset = round(start * audio.samplerate)
        count = round(seconds * audio.samplerate)
        if offset >= len(audio) or count < 1:
            raise ValueError(f"Empty requested audio window: {path}")
        audio.seek(offset)
        values = audio.read(count, dtype="float32", always_2d=True)
        if len(values) < count and not pad:
            raise ValueError(f"Truncated audio window: {path}")
        values = values.mean(axis=1)
        if audio.samplerate != sample_rate:
            divisor = gcd(audio.samplerate, sample_rate)
            values = resample_poly(values, sample_rate // divisor, audio.samplerate // divisor)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite decoded audio: {path}")
    size = round(seconds * sample_rate)
    if size < 1:
        raise ValueError("Requested duration is shorter than one output sample.")
    values = np.pad(values[:size], (0, max(0, size - len(values))))
    return torch.from_numpy(values.astype(np.float32, copy=False))


def _targets(metadata: pd.DataFrame, targets: pd.DataFrame, *, key: str) -> torch.Tensor:
    targets = align_targets(metadata, targets, key=key)
    if not targets.columns.is_unique or targets.shape[1] == 0:
        raise ValueError("Target species columns must be unique and nonempty.")
    values = feature_tensor(targets, targets.columns.tolist())
    if not ((values == 0) | (values == 1)).all():
        raise ValueError("Audio targets must be binary.")
    return values


class FocalAudioDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Random training crops or centered validation crops with weak clip targets.

    torch RNG follows DataLoader worker seeds. Short clips are right-padded;
    even a non-silent crop need not contain every weakly labeled species.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        targets: pd.DataFrame,
        *,
        training: bool,
        transform: Transform | None = None,
    ) -> None:
        self.paths = metadata.path.tolist()
        self.targets = _targets(metadata, targets, key="filename")
        self.training = training
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: accelerate for GPU training (reuse headers and profile CPU preprocessing).
        info = sf.info(self.paths[index])
        available = max(0, info.frames - SEGMENT_SECONDS * info.samplerate)
        offset = int(torch.randint(available + 1, ()).item()) if self.training else available // 2
        waveform = read_audio_window(self.paths[index], start=offset / info.samplerate, pad=True)
        inputs = self.transform(waveform) if self.transform else waveform
        return inputs, self.targets[index]


class SoundscapeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Exact annotated windows; no random crop, padding or unannotated negatives."""

    def __init__(
        self, metadata: pd.DataFrame, targets: pd.DataFrame, *, transform: Transform | None = None
    ) -> None:
        if not (
            (metadata.start >= 0)
            & (metadata.end <= SOUNDSCAPE_SECONDS)
            & (metadata.start % SEGMENT_SECONDS == 0)
            & (metadata.end - metadata.start == SEGMENT_SECONDS)
        ).all():
            raise ValueError("Expected aligned 5-second soundscape windows.")
        self.windows = list(zip(metadata.path, metadata.start, strict=True))
        self.targets = _targets(metadata, targets, key="row_id")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, start = self.windows[index]
        waveform = read_audio_window(path, start=start)
        inputs = self.transform(waveform) if self.transform else waveform
        return inputs, self.targets[index]
