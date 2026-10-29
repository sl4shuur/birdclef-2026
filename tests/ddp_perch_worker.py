"""Audit real two-process Perch training from an initially empty cache."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger

from birdclef.config import DataConfig, LoaderConfig, RuntimeConfig, TrainConfig
from birdclef.experiment import RunContext
from birdclef.models import (
    AudioClassifier,
    AudioClassifierModule,
    AudioExperimentConfig,
    AudioModelConfig,
)
from birdclef.models.classifier import backends
from birdclef.models.encoders.config import EncoderName
from birdclef.models.encoders.perch import PerchEncoder

ROOT = Path(sys.argv[1]).resolve()


class AuditedModule(AudioClassifierModule):
    def on_train_epoch_end(self) -> None:
        (ROOT / f"trained-rank-{self.global_rank}").touch()


def main() -> None:
    torch.set_num_threads(1)
    if sys.platform == "win32":
        os.environ.setdefault("USE_LIBUV", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    original = PerchEncoder.__call__

    def extract(self, waveforms):
        assert "LOCAL_RANK" not in os.environ, "DDP workers must only read cached features."
        with (ROOT / "extracted.txt").open("a") as stream:
            stream.write(f"{len(waveforms)}\n")
        return original(self, waveforms)

    PerchEncoder.__call__ = extract
    backends.AudioClassifierModule = AuditedModule
    config = AudioExperimentConfig(
        data=DataConfig(cache_dir=ROOT / "cache"),
        model=AudioModelConfig(
            encoder=EncoderName.PERCH,
            encoder_path=Path(sys.argv[2]),
            unfreeze_epoch=None,
            hidden_dim=4,
        ),
        runtime=RuntimeConfig(accelerator="cpu", devices=2, precision="32-true"),
        loader=LoaderConfig(batch_size=2, val_batch_size=1, predict_batch_size=2),
        train=TrainConfig(max_epochs=1, log_every_n_steps=1),
    )
    model = AudioClassifier(config, classes=["a", "b"])
    context = RunContext(ROOT / "run", CSVLogger(ROOT, name="logs"), Mock(), "test-fold")
    model.fit(pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "valid.csv"), context)


if __name__ == "__main__":
    main()
