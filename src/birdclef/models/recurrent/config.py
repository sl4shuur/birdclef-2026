"""A small recurrent head over frozen Perch features."""

from typing import Literal, Self

from pydantic import Field, model_validator

from birdclef.models.config import AudioExperimentConfig, AudioModelConfig, ModelType
from birdclef.models.encoders.config import EncoderName


class RecurrentModelConfig(AudioModelConfig):
    encoder: EncoderName = EncoderName.PERCH
    unfreeze_epoch: int | None = None
    recurrent_dim: int = Field(default=128, ge=1)
    recurrent_layers: int = Field(default=1, ge=1)
    cell: Literal["gru", "lstm"] = "gru"
    bidirectional: bool = True

    @model_validator(mode="after")
    def validate_recurrent_encoder(self) -> Self:
        if self.encoder != EncoderName.PERCH:
            raise ValueError("The recurrent baseline requires frozen Perch embeddings.")
        return self


class RecurrentExperimentConfig(AudioExperimentConfig):
    model_type: ModelType = ModelType.RECURRENT
    model: RecurrentModelConfig = Field(...)
