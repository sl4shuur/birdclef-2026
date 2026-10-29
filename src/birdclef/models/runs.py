"""The single model-family mapping for training, export and saved-run inference."""

import json
from pathlib import Path

from birdclef.models.classifier.model import AudioClassifier
from birdclef.models.config import AudioExperimentConfig, ModelType
from birdclef.models.recurrent.model import RecurrentClassifier


def model_class(model_type: ModelType) -> type[AudioClassifier]:
    return {ModelType.CLASSIFIER: AudioClassifier, ModelType.RECURRENT: RecurrentClassifier}[
        model_type
    ]


def read_model_config(path: Path) -> AudioExperimentConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    model_type = ModelType(data.get("model_type", ModelType.CLASSIFIER))
    return model_class(model_type).config_class.model_validate(data)
