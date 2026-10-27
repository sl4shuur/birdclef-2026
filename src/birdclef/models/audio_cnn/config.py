"""EfficientNet fine-tuning settings and human-named experiment folders."""

from pydantic import Field

from birdclef.config import Config, ConfigModel


class AudioModelConfig(ConfigModel):
    pretrained: bool = True
    dropout: float = Field(default=0.2, ge=0, lt=1)
    encoder_lr: float = Field(default=3e-5, gt=0, allow_inf_nan=False)
    head_lr: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    # None keeps the encoder frozen; 0 starts fine-tuning immediately.
    unfreeze_epoch: int | None = Field(default=0, ge=0)
    # EfficientNet-B0 has nine top-level feature blocks; None selects all of them.
    unfreeze_last_n: int | None = Field(default=None, ge=1, le=9)


class AudioExperimentConfig(Config):
    model: AudioModelConfig = Field(default_factory=AudioModelConfig)
    fold: int = Field(default=0, ge=0)
    smoke: bool = False

    def run_name(self, name: str) -> str:
        """Keep the supplied relative path; smoke checks live in their own subfolder."""
        if any(
            not part.strip()
            or part.endswith((" ", "."))
            or any(character in part for character in '<>:"\\|?*')
            for part in name.split("/")
        ):
            raise ValueError("Run name must be a relative path of nonempty folder names.")
        return f"{name}/smoke" if self.smoke else name
