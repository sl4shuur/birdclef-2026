"""Public package API; optional ML dependencies load when their exports are used."""

from importlib import import_module
from typing import TYPE_CHECKING

from .labels import align_targets, encode_labels
from .manifests import filter_focal_duration, focal_manifest, soundscape_manifest
from .split import group_folds, group_holdout

if TYPE_CHECKING:
    from .audio import (
        FocalAudioDataset,
        InferenceSoundscapeDataset,
        SoundscapeDataset,
        read_audio_window,
    )
    from .datamodule import TrainingDataModule, prediction_dataloader
    from .features import FeatureDataset, feature_tensor, multilabel_dataset
    from .prepared import audio_training_datasets, load_audio_split
    from .sampling import SourceWeightedRandomSampler

__all__ = [
    "align_targets",
    "encode_labels",
    "filter_focal_duration",
    "focal_manifest",
    "soundscape_manifest",
    "group_folds",
    "group_holdout",
    "FocalAudioDataset",
    "InferenceSoundscapeDataset",
    "SoundscapeDataset",
    "read_audio_window",
    "TrainingDataModule",
    "prediction_dataloader",
    "FeatureDataset",
    "feature_tensor",
    "multilabel_dataset",
    "audio_training_datasets",
    "load_audio_split",
    "SourceWeightedRandomSampler",
]

_MODULES = {
    "FocalAudioDataset": "audio",
    "InferenceSoundscapeDataset": "audio",
    "SoundscapeDataset": "audio",
    "read_audio_window": "audio",
    "TrainingDataModule": "datamodule",
    "prediction_dataloader": "datamodule",
    "FeatureDataset": "features",
    "feature_tensor": "features",
    "multilabel_dataset": "features",
    "audio_training_datasets": "prepared",
    "load_audio_split": "prepared",
    "SourceWeightedRandomSampler": "sampling",
}


def __getattr__(name: str) -> object:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
