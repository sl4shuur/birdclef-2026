"""Encoder boundary for shared classifier training and inference."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from birdclef.data import InferenceSoundscapeDataset, audio_training_datasets
from birdclef.data.sampling import MixedSourceDataset
from birdclef.experiment import RunContext
from birdclef.models.classifier.config import AudioExperimentConfig
from birdclef.models.classifier.network import AudioClassifierModule
from birdclef.models.trainer import check_fit_inputs


class ClassifierEncoder(ABC):
    artifacts: tuple[str, ...] = ()
    transform = None
    random_crop = False
    fingerprint: str | None = None

    def __init__(self, config: AudioExperimentConfig, classes: list[str]) -> None:
        self.config = config
        self.classes = classes

    def training_data(
        self, train: pd.DataFrame, valid: pd.DataFrame, context: RunContext
    ) -> tuple[MixedSourceDataset, Dataset]:
        training, validation = audio_training_datasets(
            train,
            valid,
            classes=self.classes,
            transform=self.transform,
            random_crop=self.random_crop,
        )
        columns = ["source", "filename", "path", "start", "end", "row_id", *self.classes]
        check_fit_inputs(
            self.config,
            train.assign(encoder_fingerprint=self.fingerprint),
            valid.assign(encoder_fingerprint=self.fingerprint),
            [*columns, "encoder_fingerprint"],
            context,
        )
        return training, validation

    @abstractmethod
    def module(self, *, initialize_pretrained: bool) -> AudioClassifierModule:
        """Build the trainable PyTorch model."""

    def prediction_data(self, frame: pd.DataFrame) -> Dataset:
        return InferenceSoundscapeDataset(frame, transform=self.transform)

    def restore(self, output_dir: Path) -> None:
        """Restore external encoder state, if any; PyTorch weights live in the checkpoint."""
        return None
