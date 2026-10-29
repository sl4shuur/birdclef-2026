"""An encoder, a classification head and shared full-fold validation."""

from collections.abc import Sized
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from birdclef.competition import ROW_ID
from birdclef.config import LRSchedulerConfig
from birdclef.eval import score
from birdclef.models.audio_cnn.config import AudioModelConfig
from birdclef.models.common import BirdClefModule, TrainingBatch


def classification_head(
    input_dim: int, num_classes: int, config: AudioModelConfig
) -> nn.Sequential:
    if config.hidden_dim is None:
        return nn.Sequential(nn.Dropout(config.dropout), nn.Linear(input_dim, num_classes))
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, num_classes),
    )


class AudioClassifierModule(BirdClefModule):
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        config: AudioModelConfig,
        lr_scheduler: LRSchedulerConfig,
        *,
        classes: list[str],
    ) -> None:
        super().__init__(
            learning_rate=config.head_lr,
            weight_decay=config.weight_decay,
            lr_scheduler=lr_scheduler,
        )
        self.encoder = encoder
        self.head = head
        self.classes = classes.copy()
        self.save_hyperparameters({"classes": self.classes}, logger=False)
        self._validation: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint["hyper_parameters"]["classes"] != self.classes:
            raise ValueError("Checkpoint class order differs from classes.json.")

    def on_validation_epoch_start(self) -> None:
        self._validation.clear()

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        self.log("val/loss", loss, batch_size=len(inputs), sync_dist=True, prog_bar=True)
        self._validation.append((targets.detach(), logits.sigmoid().detach().float()))

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._validation.clear()
            return
        values = torch.stack(
            [torch.cat([pair[index] for pair in self._validation]) for index in (0, 1)], dim=1
        )
        self._validation.clear()

        if self.trainer.world_size > 1:
            loader = self.trainer.val_dataloaders
            assert isinstance(loader, DataLoader)
            assert isinstance(loader.dataset, Sized)

            # DistributedSampler assigns rank::world_size and pads the tail. Restore
            # the sequential order, then discard padding before computing full-fold AUC.
            gathered = self.all_gather(values)
            assert isinstance(gathered, torch.Tensor)
            values = gathered.transpose(0, 1).flatten(0, 1)[: len(loader.dataset)]

        targets, probabilities = values.cpu().numpy().transpose(1, 0, 2)
        solution = pd.DataFrame(targets, columns=self.classes)
        submission = pd.DataFrame(probabilities, columns=self.classes)

        ids = np.arange(len(targets))
        solution.insert(0, ROW_ID, ids)
        submission.insert(0, ROW_ID, ids)

        self.log("val/auc", score(solution, submission), sync_dist=True, prog_bar=True)
        self.log("val/auc_classes", int(np.count_nonzero(targets.sum(axis=0))), sync_dist=True)
