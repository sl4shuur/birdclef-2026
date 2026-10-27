"""Lightning training/resume and single-device inference, shared between models."""

import hashlib
import json
import time
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from birdclef.config import Config, LoaderConfig, RuntimeConfig, TrainConfig
from birdclef.experiment import RunContext
from birdclef.models import (
    FitState,
    read_fit_state,
    save_fit_state,
    training_callbacks,
)
from birdclef.runtime import resolve_runtime


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
        callbacks=[
            *callbacks,
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=10),
        ],
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
        callbacks=[TQDMProgressBar(refresh_rate=10)],
    )
    batches = trainer.predict(module, loader, ckpt_path=best, weights_only=False)
    if not batches:
        raise ValueError("Prediction requires at least one batch.")
    tensors: list[torch.Tensor] = []
    for batch in batches:
        if not isinstance(batch, torch.Tensor):
            raise TypeError("Single-loader prediction must return tensor batches.")
        tensors.append(batch.detach().cpu())
    return torch.cat(tensors)


def check_fit_inputs(
    config: Config,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: list[str],
    context: RunContext,
    *,
    classes: list[str] | None = None,
) -> None:
    """Shared resume guard: only epoch budget and checkpoint count may change."""
    output_dir = context.output_dir
    settings = config.safe_dump()
    settings = {key: settings[key] for key in ("model", "loader", "lr_scheduler", "train")}
    for key in ("max_epochs", "checkpoint_top_k"):
        settings["train"].pop(key)
    checksums = [
        hashlib.sha256(
            pd.util.hash_pandas_object(frame[columns], index=True).to_numpy().tobytes()
        ).hexdigest()
        for frame in (train, valid)
    ]
    signature = {"settings": settings, "data": checksums, "protocol": context.protocol}
    path = output_dir / "fit-inputs.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if classes is not None:
            legacy_classes = previous["settings"]["model"].pop("target_columns", None)
            if legacy_classes is not None and legacy_classes != classes:
                raise ValueError("Class order changed; choose a new run directory.")
        if previous != signature:
            raise ValueError("Data or model settings changed; choose a new run directory.")
    elif (output_dir / "checkpoints").exists() or (output_dir / "fit.json").exists():
        raise ValueError("Existing checkpoints require fit-inputs.json to resume safely.")
    if getattr(rank_zero_only, "rank", 0) == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        temporary.replace(path)
        (output_dir / "model-config.json").write_text(
            json.dumps(config.safe_dump(), indent=2), encoding="utf-8"
        )
