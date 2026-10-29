"""Audio classification settings and human-named experiment folders."""

from typing import Self

from pydantic import Field, model_validator

from birdclef.config import Config
from birdclef.models.encoders.config import EncoderConfig, EncoderName


class AudioModelConfig(EncoderConfig):
    # None selects a linear head; an integer adds one hidden layer.
    hidden_dim: int | None = Field(default=None, ge=1)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    encoder_lr: float = Field(default=3e-5, gt=0, allow_inf_nan=False)
    head_lr: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    # None keeps the encoder frozen; 0 starts fine-tuning immediately.
    unfreeze_epoch: int | None = Field(default=0, ge=0)
    # None selects all encoder blocks; the encoder determines the upper bound.
    unfreeze_last_n: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_encoder(self) -> Self:
        if self.encoder == EncoderName.PERCH:
            if self.encoder_path is None:
                raise ValueError("Perch requires model.encoder_path pointing to a local ONNX file.")
            if self.unfreeze_epoch is not None:
                raise ValueError("Perch is frozen; set model.unfreeze_epoch=None.")
            if self.unfreeze_last_n is not None:
                raise ValueError("Perch is frozen; model.unfreeze_last_n must be None.")
        return self


class AudioExperimentConfig(Config):
    model: AudioModelConfig = Field(default_factory=AudioModelConfig)
    fold: int = Field(default=0, ge=0)
    smoke: bool = False

    def run_name(self, name: str) -> str:
        """Keep the supplied relative path; smoke checks live in their own subfolder."""
        if any(
            not part
            or part in {".", ".."}
            or part != part.strip()
            or part.endswith(".")
            or any(character in part for character in '<>:"\\|?*')
            for part in name.split("/")
        ):
            raise ValueError("Run name must be a relative path of nonempty folder names.")
        return f"{name}/smoke" if self.smoke else name
