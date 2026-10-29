"""Connect audio or cached features to the single-window classifier lifecycle."""

import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from birdclef.data import FeatureDataset, InferenceSoundscapeDataset, audio_training_datasets
from birdclef.data.embeddings import embeddings
from birdclef.data.features import feature_tensor
from birdclef.data.sampling import MixedSourceDataset
from birdclef.experiment import RunContext
from birdclef.models.classifier.config import AudioExperimentConfig
from birdclef.models.classifier.network import AudioClassifierModule, classification_head
from birdclef.models.encoders.config import EncoderName
from birdclef.models.encoders.efficientnet import EfficientNetEncoder
from birdclef.models.encoders.perch import PerchEncoder
from birdclef.models.trainer import check_fit_inputs
from birdclef.processing import log_mel


class ClassifierBackend(ABC):
    artifacts: tuple[str, ...] = ()
    transform = None
    random_crop = False
    fingerprint: str | None = None
    output_dim: int | None = None

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
    def network_encoder(self, *, initialize_pretrained: bool) -> nn.Module:
        """Return the PyTorch part and set the head's input dimension."""

    def module(self, *, initialize_pretrained: bool) -> AudioClassifierModule:
        encoder = self.network_encoder(initialize_pretrained=initialize_pretrained)
        assert self.output_dim is not None
        return AudioClassifierModule(
            encoder,
            classification_head(self.output_dim, len(self.classes), self.config.model),
            self.config.model,
            self.config.lr_scheduler,
            classes=self.classes,
        )

    def prediction_data(self, frame: pd.DataFrame) -> Dataset:
        return InferenceSoundscapeDataset(frame, transform=self.transform)

    def restore(self, output_dir: Path) -> None:
        """External state only; PyTorch weights are restored from the checkpoint."""
        return None


class EfficientNetBackend(ClassifierBackend):
    transform = staticmethod(log_mel)
    random_crop = True

    def network_encoder(self, *, initialize_pretrained: bool) -> EfficientNetEncoder:
        encoder = EfficientNetEncoder(
            pretrained=self.config.model.pretrained and initialize_pretrained
        )
        self.output_dim = encoder.output_dim
        return encoder


class PerchBackend(ClassifierBackend):
    artifacts = ("encoder.onnx", "encoder.json")

    def training_data(
        self, train: pd.DataFrame, valid: pd.DataFrame, context: RunContext
    ) -> tuple[MixedSourceDataset, Dataset]:
        assert self.config.model.encoder_path is not None
        self.encoder = PerchEncoder(
            self.config.model.encoder_path, device=self.config.model.encoder_device
        )
        self.fingerprint = self.encoder.sha256
        self.cache_dir = self.config.data.cache_dir / "perch-center-v1" / self.encoder.sha256
        training, validation = super().training_data(train, valid, context)
        datasets = {
            source: self._cached_dataset(train.loc[train.source.eq(source)], dataset)
            for source, dataset in training.by_source.items()
        }
        validation = self._cached_dataset(valid, validation)
        self._save_encoder(context.output_dir)
        return MixedSourceDataset(datasets), validation

    def restore(self, output_dir: Path) -> None:
        metadata = json.loads((output_dir / "encoder.json").read_text(encoding="utf-8"))
        # Saved runs always support offline CPU inference, even after GPU extraction.
        self.encoder = PerchEncoder(output_dir / "encoder.onnx")
        if self.encoder.sha256 != metadata["sha256"]:
            raise ValueError("Saved Perch encoder checksum mismatch.")
        self.output_dim = metadata["embedding_dim"]
        self.cache_dir = None

    def network_encoder(self, *, initialize_pretrained: bool) -> nn.Identity:
        return nn.Identity()

    def prediction_data(self, frame: pd.DataFrame) -> FeatureDataset:
        return FeatureDataset(self._features(frame, InferenceSoundscapeDataset(frame)))

    def _features(
        self, frame: pd.DataFrame, dataset: Dataset, *, cache_only: bool = False
    ) -> torch.Tensor:
        inputs = embeddings(
            frame,
            dataset,
            self.encoder,
            loader=self.config.loader,
            cache_dir=self.cache_dir,
            cache_only=cache_only,
        )
        if self.output_dim is not None and inputs.shape[1] != self.output_dim:
            raise ValueError("Encoder and cached embedding dimensions differ.")
        self.output_dim = inputs.shape[1]
        return inputs

    def _cached_dataset(self, frame: pd.DataFrame, dataset: Dataset) -> TensorDataset:
        # Lightning's parent prepares the cache before launching workers. External
        # launchers (e.g. torchrun) must receive a complete cache from a prior run.
        inputs = self._features(frame, dataset, cache_only="LOCAL_RANK" in os.environ)
        return TensorDataset(inputs, feature_tensor(frame, self.classes))

    @rank_zero_only
    def _save_encoder(self, output_dir: Path) -> None:
        destination = output_dir / "encoder.onnx"
        if self.encoder.path.resolve() != destination.resolve():
            shutil.copyfile(self.encoder.path, destination)
        metadata = {"sha256": self.encoder.sha256, "embedding_dim": self.output_dim}
        (output_dir / "encoder.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def backend_class(name: EncoderName) -> type[ClassifierBackend]:
    return {
        EncoderName.EFFICIENTNET: EfficientNetBackend,
        EncoderName.PERCH: PerchBackend,
    }[name]
