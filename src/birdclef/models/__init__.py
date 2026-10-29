"""Public package API; optional ML dependencies load when their exports are used."""

from importlib import import_module
from typing import TYPE_CHECKING

from .classifier.config import AudioExperimentConfig, AudioModelConfig, EncoderName

if TYPE_CHECKING:
    from .checkpoints import FitState, read_fit_state, save_fit_state, training_callbacks
    from .classifier.audio_cnn.network import AudioCNNModule
    from .classifier.encoder import ClassifierEncoder
    from .classifier.model import AudioClassifier
    from .classifier.network import AudioClassifierModule
    from .common import BirdClefModule, TrainingBatch
    from .trainer import check_fit_inputs, fit_module, predict_module

__all__ = [
    "AudioExperimentConfig",
    "AudioModelConfig",
    "EncoderName",
    "AudioCNNModule",
    "AudioClassifier",
    "AudioClassifierModule",
    "ClassifierEncoder",
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
    "AudioCNNModule": "classifier.audio_cnn.network",
    "AudioClassifier": "classifier.model",
    "AudioClassifierModule": "classifier.network",
    "ClassifierEncoder": "classifier.encoder",
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
