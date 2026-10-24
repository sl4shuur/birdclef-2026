"""Datasets for precomputed features; independent of the model architecture.

Dataset preparation owns inputs, target granularity and conversion to tensors.
Focal clip labels are weak labels; soundscape windows need interval-level targets.
These helpers do not decode audio, crop recordings or expand clip labels to windows.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset


def feature_tensor(frame: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    """Return finite float32 inputs in the supplied row and feature order."""
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Features and targets must be finite numeric values.")
    return torch.from_numpy(values)


class FeatureDataset(Dataset[torch.Tensor]):
    """Inference samples contain inputs only, without a singleton tuple wrapper."""

    def __init__(self, frame: pd.DataFrame, columns: list[str]) -> None:
        self.features = feature_tensor(frame, columns)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]


def multilabel_dataset(
    frame: pd.DataFrame, *, feature_columns: list[str], target_columns: list[str]
) -> TensorDataset:
    """Each training sample is (inputs, binary targets), both one-dimensional.

    Target columns follow the sample submission's species order. A row represents
    the labeled unit prepared by the caller: a focal clip or a soundscape window.
    """
    inputs = feature_tensor(frame, feature_columns)
    targets = feature_tensor(frame, target_columns)
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Multilabel training targets must be binary.")
    return TensorDataset(inputs, targets)
