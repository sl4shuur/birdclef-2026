"""Trainable EfficientNet-B0: mono Log-Mel spectrograms to feature vectors."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetEncoder(nn.Sequential):
    def __init__(self, *, pretrained: bool = True) -> None:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        stem = backbone.features[0]
        assert isinstance(stem, nn.Sequential)
        original = stem[0]
        assert isinstance(original, nn.Conv2d)
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
        super().__init__(*backbone.features.children())
        self.output_dim = classifier.in_features
        self.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("Expected mono Log-Mel inputs shaped (batch, 1, frequency, time).")
        # Per-example normalization never fits statistics to the validation set.
        mean = inputs.mean(dim=(-2, -1), keepdim=True)
        std = inputs.std(dim=(-2, -1), keepdim=True, correction=0).clamp_min(1e-6)
        inputs = ((inputs - mean) / std).contiguous(memory_format=torch.channels_last)
        features = super().forward(inputs)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)
