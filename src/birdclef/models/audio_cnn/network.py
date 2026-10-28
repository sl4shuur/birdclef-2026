"""EfficientNet-B0 encoder and a separate global-pooling classification head."""

from collections.abc import Sized
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from birdclef.competition import ROW_ID
from birdclef.config import LRSchedulerConfig
from birdclef.eval import score
from birdclef.models import AudioModelConfig, BirdClefModule, TrainingBatch


class AudioCNNModule(BirdClefModule):
    def __init__(
        self,
        config: AudioModelConfig,
        lr_scheduler: LRSchedulerConfig,
        *,
        classes: list[str],
        initialize_pretrained: bool = True,
    ) -> None:
        super().__init__(
            learning_rate=config.head_lr,
            weight_decay=config.weight_decay,
            lr_scheduler=lr_scheduler,
        )
        weights = (
            EfficientNet_B0_Weights.IMAGENET1K_V1
            if config.pretrained and initialize_pretrained
            else None
        )
        backbone = efficientnet_b0(weights=weights)
        stem = backbone.features[0]
        if not isinstance(stem, nn.Sequential):
            raise TypeError("Expected a sequential EfficientNet stem.")
        original = stem[0]
        if not isinstance(original, nn.Conv2d):
            raise TypeError("Expected an EfficientNet convolutional stem.")
        mono = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=(original.kernel_size[0], original.kernel_size[1]),
            stride=(original.stride[0], original.stride[1]),
            padding=original.padding
            if isinstance(original.padding, str)
            else (original.padding[0], original.padding[1]),
            bias=False,
        )
        with torch.no_grad():
            mono.weight.copy_(original.weight.mean(dim=1, keepdim=True))
        stem[0] = mono
        self.encoder = backbone.features
        self.encoder.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(1280, len(classes)))
        self.config = config
        self.classes = classes.copy()
        self.save_hyperparameters({"classes": self.classes}, logger=False)
        self._validation: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._set_encoder_trainability()

    def _set_encoder_trainability(self) -> None:
        epoch = self.config.unfreeze_epoch
        unfrozen = epoch is not None and self.current_epoch >= epoch
        first = len(self.encoder) - (self.config.unfreeze_last_n or len(self.encoder))
        self.encoder.train(self.training and unfrozen)
        for index, block in enumerate(self.encoder):
            trainable = unfrozen and index >= first
            block.requires_grad_(trainable)
            # Frozen blocks keep BatchNorm statistics and stochastic depth fixed.
            block.train(self.training and trainable)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self._set_encoder_trainability()
        return self

    def on_train_epoch_start(self) -> None:
        self._set_encoder_trainability()

    def optimizer_groups(self) -> list[dict[str, Any]]:
        # Include frozen parameters so unfreezing never changes optimizer/scheduler state.
        return [
            {"params": self.encoder.parameters(), "lr": self.config.encoder_lr, "name": "encoder"},
            {"params": self.head.parameters(), "lr": self.config.head_lr, "name": "head"},
        ]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint["hyper_parameters"]["classes"] != self.classes:
            raise ValueError("Checkpoint class order differs from classes.json.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1:3] != (1, 128):
            raise ValueError("Expected Log-Mel inputs shaped (B, 1, 128, time).")
        # Per-example standardization: no statistics fitted to validation data.
        mean = inputs.mean(dim=(-2, -1), keepdim=True)
        std = inputs.std(dim=(-2, -1), keepdim=True, correction=0).clamp_min(1e-6)
        inputs = ((inputs - mean) / std).contiguous(memory_format=torch.channels_last)
        features = self.encoder(inputs)
        return self.head(self.pool(features).flatten(1))

    def on_validation_epoch_start(self) -> None:
        self._validation.clear()

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        self.log("val/loss", loss, batch_size=len(inputs), sync_dist=True)
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
        self.log("val/auc", score(solution, submission), sync_dist=True)
        self.log("val/auc_classes", int(np.count_nonzero(targets.sum(axis=0))), sync_dist=True)
