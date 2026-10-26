"""EfficientNet-B0 encoder and a separate global-pooling classification head."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from birdclef.competition import ROW_ID
from birdclef.config import LRSchedulerConfig
from birdclef.eval import evaluate
from birdclef.models import AudioModelConfig, BirdClefModule, TrainingBatch


class AudioCNNModule(BirdClefModule):
    def __init__(
        self,
        config: AudioModelConfig,
        lr_scheduler: LRSchedulerConfig,
        *,
        initialize_pretrained: bool = True,
    ) -> None:
        super().__init__(
            learning_rate=config.learning_rate,
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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(1280, len(config.target_columns))
        )
        self.classes = config.target_columns.copy()
        self._validation: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1:3] != (1, 128):
            raise ValueError("Expected Log-Mel inputs shaped (B, 1, 128, time).")
        # Per-example standardization: no statistics fitted to validation data.
        mean = inputs.mean(dim=(-2, -1), keepdim=True)
        std = inputs.std(dim=(-2, -1), keepdim=True, correction=0).clamp_min(1e-6)
        features = self.encoder((inputs - mean) / std)
        return self.head(self.pool(features).flatten(1))

    def loss(self, batch: TrainingBatch) -> torch.Tensor:
        inputs, targets = batch
        return nn.functional.binary_cross_entropy_with_logits(self(inputs), targets)

    def predict_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs).sigmoid()

    def on_validation_epoch_start(self) -> None:
        self._validation.clear()

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self(inputs)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        self.log("val/loss", loss, batch_size=len(inputs))
        self._validation.append((targets.detach().cpu(), logits.sigmoid().detach().float().cpu()))

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._validation.clear()
            return
        targets = torch.cat([pair[0] for pair in self._validation]).numpy()
        probabilities = torch.cat([pair[1] for pair in self._validation]).numpy()
        self._validation.clear()
        solution = pd.DataFrame(targets, columns=self.classes)
        submission = pd.DataFrame(probabilities, columns=self.classes)
        ids = np.arange(len(targets))
        solution.insert(0, ROW_ID, ids)
        submission.insert(0, ROW_ID, ids)
        auc, report = evaluate(solution, submission)
        self.log("val/auc", auc)
        self.log("val/auc_classes", int(report.auc.notna().sum()))
