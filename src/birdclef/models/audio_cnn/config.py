"""One acoustic baseline; ablation policies can be added after measurements."""

from pydantic import Field

from birdclef.config import Config, ConfigModel


class AudioModelConfig(ConfigModel):
    pretrained: bool = True
    dropout: float = Field(default=0.2, ge=0, lt=1)
    learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)

    def run_label(self) -> str:
        initialization = "imagenet" if self.pretrained else "scratch"
        return f"effb0-{initialization}-drop{self.dropout:g}"


class AudioExperimentConfig(Config):
    model: AudioModelConfig = Field(default_factory=AudioModelConfig)
    fold: int = Field(default=0, ge=0)
    smoke: bool = False

    def run_name(self, name: str = "audio-cnn") -> str:
        """Compose model, split and training labels; the fit signature guards identity."""
        if (
            not name.strip()
            or name in {".", ".."}
            or any(character in name for character in '<>:"/\\|?*')
            or name.endswith((" ", "."))
        ):
            raise ValueError("Run name must be a nonempty folder name without path separators.")
        training = self.train.run_label(
            batch_size=self.loader.batch_size, learning_rate=self.model.learning_rate
        )
        parts = [
            name,
            f"fold{self.fold}",
            self.model.run_label(),
            training,
            f"wd{self.model.weight_decay:g}-sc{self.loader.soundscape_probability:g}",
            self.lr_scheduler.name,
        ]
        if self.smoke:
            parts.append("smoke")
        return "__".join(parts)
