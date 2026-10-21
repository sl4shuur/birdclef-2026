"""Example model settings; shared training/runtime settings come from Config."""

from pydantic import Field, model_validator

from birdclef.config import Config, ConfigModel


class ModelConfig(ConfigModel):
    feature_columns: list[str] = Field(default_factory=lambda: ["feature"], min_length=1)
    target_column: str = "target"
    hidden_dim: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    learning_rate: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_columns(self) -> "ModelConfig":
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Feature columns must be unique.")
        if self.target_column in self.feature_columns:
            raise ValueError("The target column must not be used as an input feature.")
        return self


class BaselineExperimentConfig(Config):
    model: ModelConfig = Field(default_factory=ModelConfig)
