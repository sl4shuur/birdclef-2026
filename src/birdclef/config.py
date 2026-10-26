"""Central typed settings. Change .env rather than editing library code."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.table import Table
from rich.text import Text


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "birdclef").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _project_root()
ProjectPath = Annotated[Path, AfterValidator(lambda p: p if p.is_absolute() else PROJECT_ROOT / p)]


class ConfigModel(BaseModel):
    """Configuration values share one console presentation."""

    # A setting that was renamed or moved must fail loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")

    def show(self) -> None:
        table = Table(title=type(self).__name__, header_style="bold cyan", border_style="dim")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value")
        values = self.model_dump()
        for index, (name, value) in enumerate(values.items()):
            _add_config_rows(table, name, value)
            if isinstance(value, dict) and index < len(values) - 1:
                table.add_section()
        Console().print(table)


def _add_config_rows(table: Table, name: str, value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _add_config_rows(table, f"{name}.{key}", nested)
    else:
        # Text keeps user-provided paths and strings from being interpreted as Rich markup.
        table.add_row(name, Text(str(value)))


class LoggingConfig(ConfigModel):
    directory: ProjectPath = Field(default=Path("logs"), validate_default=True)
    level: str = "INFO"
    formatter: Literal["ColoredFormatter", "ContextualColorFormatter"] = "ColoredFormatter"
    date_format: str = "%d-%m-%Y %H:%M:%S"
    full_color: bool = True
    include_function: bool = True
    success_level: int = Field(default=69, ge=1)


class DataConfig(ConfigModel):
    competition_dir: ProjectPath = Field(default=Path("data/competition"), validate_default=True)
    # Runs are written to a subfolder named after the run.
    output_dir: ProjectPath = Field(default=Path("outputs"), validate_default=True)


class SplitConfig(ConfigModel):
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    seed: int = Field(default=42, ge=0)
    group_column: str | None = None
    stratify: bool = False


class TrackingConfig(ConfigModel):
    project: str = "birdclef-2026"
    entity: str | None = None
    offline: bool = True
    api_key: SecretStr | None = None


class LoaderConfig(ConfigModel):
    batch_size: int = Field(default=128, ge=1)
    val_batch_size: int = Field(default=32, ge=1)
    predict_batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    seed: int = Field(default=0, ge=0)
    # For SourceWeightedRandomSampler with focal and soundscape datasets.
    soundscape_probability: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


class LRSchedulerConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["none", "plateau", "cosine"] = "none"
    # Both schedulers advance once per epoch; cosine's horizon stays fixed on resume.
    t_max: int = Field(default=10, ge=1)
    min_lr: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    factor: float = Field(default=0.1, gt=0, lt=1)
    patience: int = Field(default=2, ge=0)
    monitor: str = "val/loss"
    mode: Literal["min", "max"] = "min"


class TrainConfig(ConfigModel):
    max_epochs: int = Field(default=10, ge=1)
    seed: int = Field(default=42, ge=0)
    checkpoint_metric: str = "val/loss"
    checkpoint_mode: Literal["min", "max"] = "min"
    checkpoint_top_k: int = Field(default=1, ge=1)
    early_stopping_patience: int = Field(default=3, ge=0)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    gradient_clip_val: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    log_every_n_steps: int = Field(default=10, ge=1)


class RuntimeConfig(ConfigModel):
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | Literal["auto"] = "auto"
    precision: Literal["auto", "32-true", "16-mixed", "bf16-mixed"] = "auto"


class Config(BaseSettings, ConfigModel):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="BCLEF_",
        env_nested_delimiter="__",
        extra="ignore",
    )
    app_name: str = "birdclef"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def safe_dump(self) -> dict:
        """Never write auth tokens into experiment artifacts."""
        return self.model_dump(mode="json", exclude={"tracking": {"api_key"}})


def get_config[C: Config](
    config_class: type[C] = Config, *, verbose: bool = True, **sections: Any
) -> C:
    """The settings from the environment; explicit `sections` take precedence over it.

    Calls without sections reuse one cached instance per class, calls with sections build a new one.
    """
    config: C = config_class(**sections) if sections else _default_config(config_class)
    if verbose:
        config.show()
    return config


@lru_cache
def _default_config[C: Config](config_class: type[C]) -> C:
    print(f"Loading default {config_class.__name__}...")
    return config_class()


if __name__ == "__main__":
    print(f"{PROJECT_ROOT=}")
    get_config(verbose=True)
