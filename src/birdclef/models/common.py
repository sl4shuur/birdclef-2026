"""Shared optimization and loss logging; subclasses own the architecture and task."""

from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from birdclef.config import LRSchedulerConfig

type TrainingBatch = tuple[torch.Tensor, torch.Tensor]


class BirdClefModule(L.LightningModule, ABC):
    """Multilabel defaults: logits, BCE loss and sigmoid prediction.

    Default batches are (inputs, targets); prediction batches are input tensors.
    Override loss/prediction or optimizer groups for model-specific behavior.
    Log additional metrics with self.log; Trainer owns the experiment logger.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        weight_decay: float = 0.01,
        lr_scheduler: LRSchedulerConfig | None = None,
    ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._lr_scheduler = lr_scheduler or LRSchedulerConfig()
        self.criterion = torch.nn.BCEWithLogitsLoss()

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one logit per class and example."""

    def loss(self, batch: TrainingBatch) -> torch.Tensor:
        inputs, targets = batch
        return self.criterion(self(inputs), targets)

    def predict_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs).sigmoid()

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> torch.Tensor:
        loss = self.loss(batch)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch[0]),
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> None:
        self.log("val/loss", self.loss(batch), batch_size=len(batch[0]), sync_dist=True)

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.predict_batch(batch).detach().cpu()

    def optimizer_groups(self) -> list[dict[str, Any]]:
        return [{"params": self.parameters()}]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.optimizer_groups(), lr=self._learning_rate, weight_decay=self._weight_decay
        )
        config = self._lr_scheduler
        if config.name == "none":
            return optimizer
        if config.name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.mode,
                factor=config.factor,
                patience=config.patience,
                min_lr=config.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": config.monitor,
                },
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config.t_max,
                    eta_min=config.min_lr,
                ),
                "interval": "epoch",
            },
        }
