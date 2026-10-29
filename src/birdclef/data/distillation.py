"""Pair a centered audio crop with the cached teacher vector for that exact crop."""

from collections.abc import Sized

import torch
from torch.utils.data import Dataset

from birdclef.processing import log_mel


class DistillationDataset(Dataset):
    def __init__(self, audio: Dataset, teacher_features: torch.Tensor) -> None:
        assert isinstance(audio, Sized)
        if len(audio) != len(teacher_features):
            raise ValueError("Audio and teacher features must have matching sample counts.")
        self.audio = audio
        self.teacher_features = teacher_features

    def __len__(self) -> int:
        return len(self.teacher_features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, _ = self.audio[index]
        return log_mel(waveform), self.teacher_features[index]
