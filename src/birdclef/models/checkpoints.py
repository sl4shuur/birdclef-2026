"""Best/last checkpoints and the training completion record, independent of the task."""

from pathlib import Path
from typing import Literal, NamedTuple, Self

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from pydantic import BaseModel, Field

from birdclef.config import TrainConfig


class TrainingCallbacks(NamedTuple):
    best: ModelCheckpoint
    last: ModelCheckpoint
    stopping: EarlyStopping


def training_callbacks(directory: Path, config: TrainConfig) -> TrainingCallbacks:
    best = ModelCheckpoint(
        dirpath=directory,
        monitor=config.checkpoint_metric,
        mode=config.checkpoint_mode,
        save_top_k=config.checkpoint_top_k,
        filename="epoch{epoch:03d}",
        auto_insert_metric_name=False,
    )
    last = ModelCheckpoint(
        dirpath=directory,
        filename="last",
        save_top_k=1,
        enable_version_counter=False,
    )
    stopping = EarlyStopping(
        monitor=config.checkpoint_metric,
        mode=config.checkpoint_mode,
        patience=config.early_stopping_patience,
    )
    return TrainingCallbacks(best, last, stopping)


class FitState(BaseModel):
    """fit.json is separate from the experiment's identity/lifecycle state."""

    trained_epochs: int = Field(default=0, ge=0)
    stop_reason: Literal["max_epochs", "early_stopping"] | None = None
    best_checkpoint: str | None = None

    @classmethod
    def from_training(
        cls,
        trainer: L.Trainer,
        callbacks: TrainingCallbacks,
        output_dir: Path,
    ) -> Self:
        best = Path(callbacks.best.best_model_path)
        if not best.is_file():
            raise FileNotFoundError("Training did not produce a best checkpoint.")
        return cls(
            trained_epochs=trainer.current_epoch,
            stop_reason="early_stopping" if callbacks.stopping.stopped_epoch > 0 else "max_epochs",
            best_checkpoint=best.resolve().relative_to(output_dir.resolve()).as_posix(),
        )

    def needs_training(self, max_epochs: int) -> bool:
        return self.stop_reason is None or (
            self.stop_reason == "max_epochs" and self.trained_epochs < max_epochs
        )

    def best_path(self, output_dir: Path) -> Path:
        if self.best_checkpoint is None:
            raise ValueError("Completed training must identify its best checkpoint.")
        path = output_dir / self.best_checkpoint
        if not path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("The best checkpoint must stay inside the run folder.")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def read_fit_state(output_dir: Path) -> FitState:
    path = output_dir / "fit.json"
    return (
        FitState.model_validate_json(path.read_text(encoding="utf-8"))
        if path.exists()
        else FitState()
    )


@rank_zero_only
def save_fit_state(state: FitState, output_dir: Path) -> None:
    temporary = output_dir / "fit.json.tmp"
    temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_dir / "fit.json")
