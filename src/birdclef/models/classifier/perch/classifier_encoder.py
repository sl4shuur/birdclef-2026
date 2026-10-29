"""Frozen Perch data, cache and saved encoder state."""

import json
import shutil
from pathlib import Path

import pandas as pd
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
from torch.utils.data import ConcatDataset, Dataset, TensorDataset

from birdclef.data import FeatureDataset, InferenceSoundscapeDataset
from birdclef.data.features import feature_tensor
from birdclef.experiment import RunContext
from birdclef.models.classifier.encoder import ClassifierEncoder
from birdclef.models.classifier.network import AudioClassifierModule, classification_head
from birdclef.models.classifier.perch.cache import embeddings
from birdclef.models.classifier.perch.encoder import PerchEncoder


class FrozenPerchEncoder(ClassifierEncoder):
    artifacts = ("encoder.onnx", "encoder.json")
    embedding_dim: int | None = None

    def training_data(
        self, train: pd.DataFrame, valid: pd.DataFrame, context: RunContext
    ) -> tuple[ConcatDataset, Dataset]:
        assert self.config.model.encoder_path is not None
        self.encoder = PerchEncoder(self.config.model.encoder_path)
        self.fingerprint = self.encoder.sha256
        self.cache_dir = self.config.data.cache_dir
        training, validation = super().training_data(train, valid, context)
        training, validation = self._cached_datasets(train, valid, training, validation)
        self._save_encoder(context.output_dir)
        return training, validation

    def restore(self, output_dir: Path) -> None:
        metadata = json.loads((output_dir / "encoder.json").read_text(encoding="utf-8"))
        self.encoder = PerchEncoder(output_dir / "encoder.onnx")
        if self.encoder.sha256 != metadata["sha256"]:
            raise ValueError("Saved Perch encoder checksum mismatch.")
        self.embedding_dim = metadata["embedding_dim"]
        self.cache_dir = None

    def module(self, *, initialize_pretrained: bool) -> AudioClassifierModule:
        assert self.embedding_dim is not None
        return AudioClassifierModule(
            nn.Identity(),
            classification_head(self.embedding_dim, len(self.classes), self.config.model),
            self.config.model,
            self.config.lr_scheduler,
            classes=self.classes,
        )

    def prediction_data(self, frame: pd.DataFrame) -> Dataset:
        inputs = embeddings(
            frame,
            InferenceSoundscapeDataset(frame),
            self.encoder,
            loader=self.config.loader,
            cache_dir=self.cache_dir,
        )
        return FeatureDataset(inputs)

    def _cached_datasets(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        training: ConcatDataset,
        validation: Dataset,
    ) -> tuple[ConcatDataset, TensorDataset]:
        datasets = []
        frames = [train.loc[train.source.eq(source)] for source in ("focal", "soundscape")]
        for frame, dataset in zip([*frames, valid], [*training.datasets, validation], strict=True):
            inputs = embeddings(
                frame, dataset, self.encoder, loader=self.config.loader, cache_dir=self.cache_dir
            )
            if self.embedding_dim is not None and inputs.shape[1] != self.embedding_dim:
                raise ValueError("Training and validation embedding dimensions differ.")
            self.embedding_dim = inputs.shape[1]
            datasets.append(TensorDataset(inputs, feature_tensor(frame, self.classes)))
        return ConcatDataset(datasets[:2]), datasets[2]

    @rank_zero_only
    def _save_encoder(self, output_dir: Path) -> None:
        destination = output_dir / "encoder.onnx"
        if self.encoder.path.resolve() != destination.resolve():
            shutil.copyfile(self.encoder.path, destination)
        metadata = {"sha256": self.encoder.sha256, "embedding_dim": self.embedding_dim}
        (output_dir / "encoder.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
