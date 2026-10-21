"""Reusable loaders; datasets and sample transformations belong to the competition."""

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from birdclef.config import LoaderConfig


class TrainingDataModule(L.LightningDataModule):
    def __init__(self, train: Dataset, valid: Dataset, *, config: LoaderConfig) -> None:
        super().__init__()
        self.train = train
        self.valid = valid
        self.config = config

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )


def prediction_dataloader(dataset: Dataset, *, config: LoaderConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.predict_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
