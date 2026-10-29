"""Settings for Perch feature pretraining; fine-tuning uses AudioExperimentConfig."""

from typing import Self

from pydantic import model_validator

from birdclef.config import ProjectPath
from birdclef.models.config import AudioExperimentConfig, ModelType
from birdclef.models.encoders.config import EncoderName


class DistillationConfig(AudioExperimentConfig):
    teacher_path: ProjectPath

    @model_validator(mode="after")
    def validate_feature_training(self) -> Self:
        if (
            self.model_type != ModelType.CLASSIFIER
            or self.model.encoder != EncoderName.EFFICIENTNET
        ):
            raise ValueError("Feature distillation requires an EfficientNet student.")
        if self.model.unfreeze_epoch != 0 or self.model.unfreeze_last_n is not None:
            raise ValueError("Feature distillation trains the entire student encoder from epoch 0.")
        if self.train.checkpoint_metric != "val/loss" or self.train.checkpoint_mode != "min":
            raise ValueError("Feature distillation selects checkpoints by minimum val/loss.")
        return self
