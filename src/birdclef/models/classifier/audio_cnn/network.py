"""EfficientNet-B0 encoder and a separate global-pooling classification head."""

from typing import Any, Self

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from birdclef.config import LRSchedulerConfig
from birdclef.models.classifier.config import AudioModelConfig, EncoderName
from birdclef.models.classifier.network import AudioClassifierModule, classification_head


class AudioCNNModule(AudioClassifierModule):
    def __init__(
        self,
        config: AudioModelConfig,
        lr_scheduler: LRSchedulerConfig,
        *,
        classes: list[str],
        initialize_pretrained: bool = True,
    ) -> None:
        if config.encoder != EncoderName.EFFICIENTNET:
            raise ValueError("AudioCNNModule requires the EfficientNet encoder.")
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
        classifier = backbone.classifier[-1]
        assert isinstance(classifier, nn.Linear)
        super().__init__(
            backbone.features,
            classification_head(classifier.in_features, len(classes), config),
            config,
            lr_scheduler,
            classes=classes,
        )
        self.encoder: nn.Sequential
        self.head: nn.Sequential
        self.encoder.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.config = config
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1:3] != (1, 128):
            raise ValueError("Expected Log-Mel inputs shaped (B, 1, 128, time).")
        # Per-example standardization: no statistics fitted to validation data.
        mean = inputs.mean(dim=(-2, -1), keepdim=True)
        std = inputs.std(dim=(-2, -1), keepdim=True, correction=0).clamp_min(1e-6)
        inputs = ((inputs - mean) / std).contiguous(memory_format=torch.channels_last)
        features = self.encoder(inputs)
        return self.head(self.pool(features).flatten(1))
