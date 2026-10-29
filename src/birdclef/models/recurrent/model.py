"""Sequence inputs with the existing fit, checkpoint and window-reporting lifecycle."""

from typing import ClassVar

import pandas as pd
import torch

from birdclef.data.sequences import recording_windows, requested_indices
from birdclef.models.classifier.model import AudioClassifier
from birdclef.models.config import AudioExperimentConfig, ModelType
from birdclef.models.recurrent.backends import RecurrentPerchBackend
from birdclef.models.recurrent.config import RecurrentExperimentConfig


class RecurrentClassifier(AudioClassifier):
    model_type: ClassVar[ModelType] = ModelType.RECURRENT
    config_class: ClassVar[type[AudioExperimentConfig]] = RecurrentExperimentConfig

    def __init__(self, config: RecurrentExperimentConfig, *, classes: list[str]) -> None:
        super().__init__(config, classes=classes, backend=RecurrentPerchBackend(config, classes))

    def smoke_training_data(self, train: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for _, part in train.groupby("source", sort=False):
            selected = part.filename.drop_duplicates().sample(
                n=min(32, part.filename.nunique()), random_state=self.config.train.seed
            )
            parts.append(part.loc[part.filename.isin(selected)])
        return pd.concat(parts, ignore_index=True)

    def _prediction_scores(self, frame: pd.DataFrame, predictions: torch.Tensor) -> torch.Tensor:
        windows = recording_windows(frame)
        return predictions.flatten(0, 1)[requested_indices(frame, windows)]
