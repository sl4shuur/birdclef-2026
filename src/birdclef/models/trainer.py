"""Lightning training/resume and single-device inference, shared between models."""

import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from birdclef.config import LoaderConfig, RuntimeConfig, TrainConfig
from birdclef.experiment.contract import RunContext
from birdclef.models.checkpoints import (
    FitState,
    read_fit_state,
    save_fit_state,
    training_callbacks,
)
from birdclef.runtime.hardware import resolve_runtime


class _CurrentStreamDDPStrategy(DDPStrategy):
    """Create gradient accumulators on the stream used by eager training.

    Workaround: https://github.com/Lightning-AI/pytorch-lightning/pull/21746
    """

    def _setup_model(self, model: torch.nn.Module) -> DistributedDataParallel:
        return DistributedDataParallel(
            module=model, device_ids=self.determine_ddp_device_ids(), **self._ddp_kwargs
        )


def fit_module(
    module: L.LightningModule,
    data: L.LightningDataModule,
    *,
    train_config: TrainConfig,
    loader_config: LoaderConfig,
    runtime: RuntimeConfig,
    run: RunContext,
) -> Path:
    """Return the best checkpoint; last.ckpt resumes optimizer, scheduler and epoch state.

    The caller must check data/configuration compatibility before reusing a run folder.
    """
    state = read_fit_state(run.output_dir)
    if not state.needs_training(train_config.max_epochs):
        run.logger.info(f"Training already finished: {state.trained_epochs} epochs")
        return state.best_path(run.output_dir)
    directory = run.output_dir / "checkpoints"
    last = directory / "last.ckpt"
    if state.trained_epochs and not last.is_file():
        raise FileNotFoundError(f"Continuing this run requires {last}")
    callbacks = training_callbacks(directory, train_config)
    resolved = resolve_runtime(runtime)
    devices = resolved.devices
    if devices == "auto":
        devices = 1 if resolved.accelerator == "cpu" else max(torch.cuda.device_count(), 1)
    trainer = L.Trainer(
        **resolved.model_dump(),
        strategy=_CurrentStreamDDPStrategy() if devices > 1 else "auto",
        max_epochs=train_config.max_epochs,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=train_config.gradient_clip_val,
        log_every_n_steps=train_config.log_every_n_steps,
        logger=run.tracker,
        callbacks=list(callbacks),
    )
    run.tracker.log_hyperparams(
        {
            "runtime": runtime.model_dump(),
            "num_devices": trainer.num_devices,
            "effective_batch_size": (
                loader_config.batch_size
                * trainer.num_devices
                * train_config.accumulate_grad_batches
            ),
        }
    )
    started = time.perf_counter()
    # Load only checkpoints from a trusted run: full resume includes optimizer state.
    trainer.fit(
        module, datamodule=data, ckpt_path=last if last.is_file() else None, weights_only=False
    )
    state = FitState.from_training(trainer, callbacks, run.output_dir)
    save_fit_state(state, run.output_dir)
    if trainer.is_global_zero:
        run.tracker.log_metrics({"perf/fit_seconds": time.perf_counter() - started})
    return state.best_path(run.output_dir)


def predict_module(
    module: L.LightningModule,
    loader: DataLoader,
    *,
    runtime: RuntimeConfig,
    best: Path,
) -> torch.Tensor:
    """Predict in loader order on one device, avoiding distributed sampler padding."""
    trainer = L.Trainer(
        **resolve_runtime(runtime).model_copy(update={"devices": 1}).model_dump(),
        logger=False,
        enable_checkpointing=False,
    )
    batches = trainer.predict(module, loader, ckpt_path=best, weights_only=False)
    if not batches:
        raise ValueError("Prediction requires at least one batch.")
    return torch.cat([batch.detach().cpu() for batch in batches])
