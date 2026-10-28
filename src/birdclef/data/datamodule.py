"""Reusable loaders; datasets and sample transformations belong to the competition."""

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from birdclef.config import LoaderConfig


class TrainingDataModule(L.LightningDataModule):
    def __init__(
        self,
        train: Dataset,
        valid: Dataset,
        *,
        config: LoaderConfig,
        sampler: Sampler[int] | None = None,
    ) -> None:
        super().__init__()
        self.train = train
        self.valid = valid
        self.config = config
        self.sampler = sampler

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.config.batch_size,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
            pin_memory=self.config.pin_memory,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
            pin_memory=self.config.pin_memory,
        )


def prediction_dataloader(dataset: Dataset, *, config: LoaderConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.predict_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=config.pin_memory,
    )
