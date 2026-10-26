"""Public package API; optional ML dependencies load when their exports are used."""

from importlib import import_module
from typing import TYPE_CHECKING

from .audio_cnn.config import AudioExperimentConfig, AudioModelConfig
from .baseline.config import BaselineExperimentConfig
from .baseline.config import ModelConfig as BaselineModelConfig

if TYPE_CHECKING:
    from .audio_cnn.model import AudioCNN
    from .audio_cnn.network import AudioCNNModule
    from .baseline.model import Baseline
    from .baseline.network import BaselineModule
    from .checkpoints import FitState, read_fit_state, save_fit_state, training_callbacks
    from .common import BirdClefModule, TrainingBatch
    from .trainer import check_fit_inputs, fit_module, predict_module

__all__ = [
    "AudioExperimentConfig",
    "AudioModelConfig",
    "BaselineExperimentConfig",
    "BaselineModelConfig",
    "AudioCNN",
    "AudioCNNModule",
    "Baseline",
    "BaselineModule",
    "FitState",
    "read_fit_state",
    "save_fit_state",
    "training_callbacks",
    "BirdClefModule",
    "TrainingBatch",
    "check_fit_inputs",
    "fit_module",
    "predict_module",
]

_MODULES = {
    "AudioCNN": "audio_cnn.model",
    "AudioCNNModule": "audio_cnn.network",
    "Baseline": "baseline.model",
    "BaselineModule": "baseline.network",
    "FitState": "checkpoints",
    "read_fit_state": "checkpoints",
    "save_fit_state": "checkpoints",
    "training_callbacks": "checkpoints",
    "BirdClefModule": "common",
    "TrainingBatch": "common",
    "check_fit_inputs": "trainer",
    "fit_module": "trainer",
    "predict_module": "trainer",
}


def __getattr__(name: str) -> object:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
