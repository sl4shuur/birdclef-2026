"""A working MLP regression example; replace the network and loss for your task."""

import torch
from torch import nn

from birdclef.config import LRSchedulerConfig
from birdclef.models.baseline.config import ModelConfig
from birdclef.models.common import KaggleModule, TrainingBatch


class BaselineModule(KaggleModule):
    def __init__(self, config: ModelConfig, lr_scheduler: LRSchedulerConfig) -> None:
        super().__init__(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_scheduler=lr_scheduler,
        )
        self.network = nn.Sequential(
            nn.Linear(len(config.feature_columns), config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)

    def loss(self, batch: TrainingBatch) -> torch.Tensor:
        inputs, targets = batch
        return nn.functional.mse_loss(self(inputs), targets)

    def predict_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs)
