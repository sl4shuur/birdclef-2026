"""Example model settings; shared training/runtime settings come from Config."""

from pydantic import Field, model_validator

from birdclef.competition import ROW_ID
from birdclef.config import Config, ConfigModel


class ModelConfig(ConfigModel):
    feature_columns: list[str] = Field(default_factory=lambda: ["feature"], min_length=1)
    # Set to class_columns(sample_submission) for a competition model.
    target_columns: list[str] = Field(default_factory=lambda: ["target"], min_length=1)
    hidden_dim: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    learning_rate: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_columns(self) -> "ModelConfig":
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Feature columns must be unique.")
        if len(set(self.target_columns)) != len(self.target_columns):
            raise ValueError("Target columns must be unique.")
        if set(self.target_columns) & set(self.feature_columns):
            raise ValueError("Target columns must not be used as input features.")
        if any(not c.strip() or c == ROW_ID for c in self.feature_columns + self.target_columns):
            raise ValueError("Feature/target names must be nonempty and must not be row_id.")
        return self


class BaselineExperimentConfig(Config):
    model: ModelConfig = Field(default_factory=ModelConfig)
