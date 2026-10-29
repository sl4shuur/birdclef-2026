"""Public package API; optional ML dependencies load when their exports are used."""

from importlib import import_module
from typing import TYPE_CHECKING

from .config import AudioExperimentConfig, AudioModelConfig, ModelType
from .encoders.config import EncoderConfig, EncoderName
from .recurrent.config import RecurrentExperimentConfig, RecurrentModelConfig

if TYPE_CHECKING:
    from .checkpoints import FitState, read_fit_state, save_fit_state, training_callbacks
    from .classifier.backends import ClassifierBackend
    from .classifier.model import AudioClassifier
    from .classifier.network import AudioClassifierModule
    from .common import BirdClefModule, TrainingBatch
    from .encoders.efficientnet import EfficientNetEncoder
    from .encoders.perch import PerchEncoder
    from .recurrent.model import RecurrentClassifier
    from .recurrent.network import RecurrentClassifierModule
    from .trainer import check_fit_inputs, fit_module, predict_module

__all__ = [
    "AudioExperimentConfig",
    "AudioModelConfig",
    "ModelType",
    "RecurrentExperimentConfig",
    "RecurrentModelConfig",
    "RecurrentClassifier",
    "RecurrentClassifierModule",
    "EncoderName",
    "EncoderConfig",
    "EfficientNetEncoder",
    "PerchEncoder",
    "AudioClassifier",
    "AudioClassifierModule",
    "ClassifierBackend",
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
    "RecurrentClassifier": "recurrent.model",
    "RecurrentClassifierModule": "recurrent.network",
    "EfficientNetEncoder": "encoders.efficientnet",
    "PerchEncoder": "encoders.perch",
    "AudioClassifier": "classifier.model",
    "AudioClassifierModule": "classifier.network",
    "ClassifierBackend": "classifier.backends",
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
