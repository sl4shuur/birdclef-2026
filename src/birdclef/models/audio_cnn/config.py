"""One acoustic baseline; ablation policies can be added after measurements."""

from pydantic import Field, field_validator

from birdclef.competition import ROW_ID
from birdclef.config import Config, ConfigModel


class AudioModelConfig(ConfigModel):
    target_columns: list[str] = Field(min_length=1)
    pretrained: bool = True
    dropout: float = Field(default=0.2, ge=0, lt=1)
    learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)

    @field_validator("target_columns")
    @classmethod
    def validate_classes(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values) or any(
            not value.strip() or value == ROW_ID for value in values
        ):
            raise ValueError("Species must be unique, nonempty and exclude row_id.")
        return values


class AudioExperimentConfig(Config):
    model: AudioModelConfig
