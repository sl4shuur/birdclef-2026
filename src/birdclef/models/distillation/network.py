"""Train a student encoder and a temporary projection using cosine distance."""

import torch
from torch import nn
from torch.nn import functional as F

from birdclef.config import LRSchedulerConfig
from birdclef.models.common import BirdClefModule, TrainingBatch
from birdclef.models.config import AudioModelConfig


class FeatureDistillationModule(BirdClefModule):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        teacher_dim: int,
        config: AudioModelConfig,
        lr_scheduler: LRSchedulerConfig,
    ) -> None:
        super().__init__(
            learning_rate=config.encoder_lr,
            weight_decay=config.weight_decay,
            lr_scheduler=lr_scheduler,
        )
        self.encoder = encoder
        self.projection = nn.Linear(input_dim, teacher_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(inputs))

    def loss(self, batch: TrainingBatch) -> torch.Tensor:
        inputs, teacher = batch
        return (1 - F.cosine_similarity(self(inputs).float(), teacher.float(), dim=-1)).mean()
