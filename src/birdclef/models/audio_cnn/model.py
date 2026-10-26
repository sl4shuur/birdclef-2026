"""Acoustic model lifecycle using the shared trainer and prepared split frames."""

import json
from pathlib import Path
from typing import Self

import lightning as L
import pandas as pd
import torch

from birdclef.competition import ROW_ID
from birdclef.data import (
    InferenceSoundscapeDataset,
    SourceWeightedRandomSampler,
    TrainingDataModule,
    audio_training_datasets,
    prediction_dataloader,
)
from birdclef.eval import evaluate
from birdclef.experiment import BirdClefModel, RunContext
from birdclef.models import (
    AudioCNNModule,
    AudioExperimentConfig,
    check_fit_inputs,
    fit_module,
    predict_module,
    read_fit_state,
)
from birdclef.processing import log_mel
from birdclef.submission import validate_row_ids


class AudioCNN(BirdClefModel):
    def __init__(self, config: AudioExperimentConfig) -> None:
        self.config = config
        self._module: AudioCNNModule | None = None
        self._best: Path | None = None
        self.sampler: SourceWeightedRandomSampler | None = None

    @classmethod
    def from_run(cls, output_dir: Path, *, config: AudioExperimentConfig | None = None) -> Self:
        saved = AudioExperimentConfig.model_validate_json(
            (output_dir / "model-config.json").read_text(encoding="utf-8")
        )
        if config is not None:
            if config.model != saved.model or config.lr_scheduler != saved.lr_scheduler:
                raise ValueError("Inference must use the saved model settings.")
            saved.runtime = config.runtime
            saved.loader = config.loader
        model = cls(saved)
        # A trained checkpoint contains every weight. Never download ImageNet at inference.
        model._module = AudioCNNModule(saved.model, saved.lr_scheduler, initialize_pretrained=False)
        model._best = read_fit_state(output_dir).best_path(output_dir)
        return model

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, context: RunContext) -> None:
        config = self.config
        if config.runtime.devices != 1:
            raise ValueError("The acoustic baseline currently supports one training device.")
        classes = config.model.target_columns
        validate_row_ids(valid)
        L.seed_everything(config.train.seed, workers=True)
        training, validation = audio_training_datasets(train, valid, classes=classes)
        # Check full-fold AUC feasibility before downloading weights or training.
        evaluate(valid[[ROW_ID, *classes]], valid[[ROW_ID, *classes]])
        columns = ["source", "filename", "path", "start", "end", ROW_ID, *classes]
        check_fit_inputs(config, train, valid, columns, context)
        self.sampler = SourceWeightedRandomSampler(
            len(training.datasets[0]),
            len(training.datasets[1]),
            soundscape_probability=config.loader.soundscape_probability,
            generator=torch.Generator().manual_seed(config.loader.seed),
        )
        data = TrainingDataModule(training, validation, config=config.loader, sampler=self.sampler)
        context.tracker.log_hyperparams(config.safe_dump())
        continuing = (context.output_dir / "checkpoints" / "last.ckpt").is_file()
        self._module = AudioCNNModule(
            config.model, config.lr_scheduler, initialize_pretrained=not continuing
        )
        self._best = fit_module(
            self._module,
            data,
            train_config=config.train,
            loader_config=config.loader,
            runtime=config.runtime,
            run=context,
        )
        predictions = self.predict(valid)
        auc, report = evaluate(valid[[ROW_ID, *classes]], predictions)
        predictions.to_csv(context.output_dir / "validation-predictions.csv", index=False)
        report.to_csv(context.output_dir / "validation-class-metrics.csv")
        metrics = {
            "auc": auc,
            "auc_classes": int(report.auc.notna().sum()),
            "windows": len(valid),
            "protocol": context.protocol,
        }
        (context.output_dir / "validation-metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        context.tracker.log_metrics({"best/val_auc": auc})
        context.tracker.save()

    def predict(self, frame: pd.DataFrame, context: RunContext | None = None) -> pd.DataFrame:
        if self._module is None or self._best is None:
            raise RuntimeError("Call fit() or from_run() before predict().")
        classes = self.config.model.target_columns
        if frame.empty:
            return pd.DataFrame(columns=[ROW_ID, *classes])
        validate_row_ids(frame)
        loader = prediction_dataloader(
            InferenceSoundscapeDataset(frame, transform=log_mel), config=self.config.loader
        )
        predictions = predict_module(
            self._module, loader, runtime=self.config.runtime, best=self._best
        )
        result = pd.DataFrame(predictions.float().numpy(), columns=classes)
        result.insert(0, ROW_ID, frame[ROW_ID].to_numpy())
        return result
