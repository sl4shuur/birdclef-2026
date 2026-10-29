"""Reuse frozen Perch extraction; assemble length-1 focal clips and full soundscapes."""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from birdclef.data import FeatureDataset, InferenceSoundscapeDataset
from birdclef.data.features import feature_tensor
from birdclef.data.sampling import MixedSourceDataset
from birdclef.data.sequences import (
    SEQUENCE_LENGTH,
    recording_windows,
    sequence_dataset,
    window_targets,
)
from birdclef.models.classifier.backends import PerchBackend
from birdclef.models.recurrent.config import RecurrentExperimentConfig
from birdclef.models.recurrent.network import RecurrentClassifierModule


class RecurrentPerchBackend(PerchBackend):
    def module(self, *, initialize_pretrained: bool) -> RecurrentClassifierModule:
        assert isinstance(self.config, RecurrentExperimentConfig)
        assert self.output_dim is not None
        return RecurrentClassifierModule(
            self.output_dim, self.config.model, self.config.lr_scheduler, classes=self.classes
        )

    def _training_features(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        training: MixedSourceDataset,
        validation: Dataset,
    ) -> tuple[MixedSourceDataset, Dataset]:
        focal = train.loc[train.source.eq("focal")]
        inputs = self._features(
            focal, training.by_source["focal"], cache_only="LOCAL_RANK" in os.environ
        )
        focal_data = sequence_dataset(
            inputs,
            feature_tensor(focal, self.classes),
            torch.ones(len(focal), dtype=torch.bool),
            length=1,
        )
        soundscape_data = self._soundscape_sequences(train.loc[train.source.eq("soundscape")])
        return MixedSourceDataset(
            {"focal": focal_data, "soundscape": soundscape_data}
        ), self._soundscape_sequences(valid)

    def _soundscape_sequences(self, labeled: pd.DataFrame) -> Dataset:
        windows = recording_windows(labeled)
        inputs = self._features(
            windows, InferenceSoundscapeDataset(windows), cache_only="LOCAL_RANK" in os.environ
        )
        targets, mask = window_targets(windows, labeled, self.classes)
        return sequence_dataset(inputs, targets, mask, length=SEQUENCE_LENGTH)

    def prediction_data(
        self, frame: pd.DataFrame, *, features: torch.Tensor | None = None
    ) -> FeatureDataset:
        if features is None:
            windows = recording_windows(frame)
            features = self._features(windows, InferenceSoundscapeDataset(windows))
        return FeatureDataset(features.reshape(-1, SEQUENCE_LENGTH, features.shape[-1]))
