"""Multilabel MLP for precomputed numeric features; no audio frontend yet."""

import torch
from torch import nn

from birdclef.config import LRSchedulerConfig
from birdclef.models import BaselineModelConfig, BirdClefModule, TrainingBatch


class BaselineModule(BirdClefModule):
    def __init__(self, config: BaselineModelConfig, lr_scheduler: LRSchedulerConfig) -> None:
        super().__init__(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_scheduler=lr_scheduler,
        )
        self.network = nn.Sequential(
            nn.Linear(len(config.feature_columns), config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, len(config.target_columns)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def loss(self, batch: TrainingBatch) -> torch.Tensor:
        inputs, targets = batch
        return nn.functional.binary_cross_entropy_with_logits(self(inputs), targets)

    def predict_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs).sigmoid()
