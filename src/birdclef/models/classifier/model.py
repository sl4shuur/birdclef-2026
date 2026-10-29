"""Acoustic model lifecycle using the shared trainer and prepared split frames."""

import json
from pathlib import Path
from typing import Self

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import Dataset

from birdclef.competition import ROW_ID, class_columns
from birdclef.data import (
    SourceWeightedRandomSampler,
    TrainingDataModule,
    prediction_dataloader,
)
from birdclef.data.sampling import MixedSourceDataset
from birdclef.eval import evaluate, score
from birdclef.experiment import BirdClefModel, RunContext
from birdclef.models import (
    AudioExperimentConfig,
    fit_module,
    predict_module,
    read_fit_state,
)
from birdclef.models.classifier.encoder import ClassifierEncoder
from birdclef.models.classifier.encoders import encoder_class
from birdclef.models.classifier.network import AudioClassifierModule
from birdclef.submission import validate_row_ids


class AudioClassifier(BirdClefModel):
    def __init__(
        self,
        config: AudioExperimentConfig,
        *,
        classes: list[str],
        encoder: ClassifierEncoder | None = None,
    ) -> None:
        self.config = config
        self.classes = class_columns(pd.DataFrame(columns=[ROW_ID, *classes]))
        self._module: AudioClassifierModule | None = None
        self._best: Path | None = None
        self.sampler: SourceWeightedRandomSampler | None = None
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_class(config.model.encoder)(config, self.classes)
        )

    @classmethod
    def from_run(cls, output_dir: Path, *, config: AudioExperimentConfig | None = None) -> Self:
        saved = AudioExperimentConfig.model_validate_json(
            (output_dir / "model-config.json").read_text(encoding="utf-8")
        )
        classes = json.loads((output_dir / "classes.json").read_text(encoding="utf-8"))
        if config is not None:
            if config.model != saved.model or config.lr_scheduler != saved.lr_scheduler:
                raise ValueError("Inference must use the saved model settings.")
            saved.runtime = config.runtime
            saved.loader = config.loader
        model = cls(saved, classes=classes)
        model.encoder.restore(output_dir)
        model._module = model.encoder.module(initialize_pretrained=False)
        model._best = read_fit_state(output_dir).inference_path(output_dir)
        return model

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, context: RunContext) -> None:
        config = self.config
        classes = self.classes
        validate_row_ids(valid)
        L.seed_everything(config.train.seed, workers=True)
        score(valid[[ROW_ID, *classes]], valid[[ROW_ID, *classes]])
        training, validation = self.encoder.training_data(train, valid, context)
        if rank_zero_only.rank == 0:
            (context.output_dir / "classes.json").write_text(
                json.dumps(classes, indent=2), encoding="utf-8"
            )
        data = self._training_loader(training, validation)
        context.tracker.log_hyperparams({**config.safe_dump(), "num_classes": len(classes)})
        continuing = (context.output_dir / "checkpoints" / "last.ckpt").is_file()
        self._module = self.encoder.module(initialize_pretrained=not continuing)
        self._best = fit_module(
            self._module,
            data,
            train_config=config.train,
            loader_config=config.loader,
            runtime=config.runtime,
            run=context,
        )
        if rank_zero_only.rank != 0:
            return
        self._save_validation(valid, context)

    def _training_loader(
        self, training: MixedSourceDataset, validation: Dataset
    ) -> TrainingDataModule:
        config = self.config
        self.sampler = SourceWeightedRandomSampler(
            training.sources,
            soundscape_probability=config.loader.soundscape_probability,
            generator=torch.Generator().manual_seed(config.loader.seed),
        )
        return TrainingDataModule(training, validation, config=config.loader, sampler=self.sampler)

    def _save_validation(self, valid: pd.DataFrame, context: RunContext) -> None:
        classes = self.classes
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
        classes = self.classes
        if frame.empty:
            return pd.DataFrame(columns=[ROW_ID, *classes])
        validate_row_ids(frame)
        loader = prediction_dataloader(
            self.encoder.prediction_data(frame), config=self.config.loader
        )
        predictions = predict_module(
            self._module, loader, runtime=self.config.runtime, best=self._best
        )
        result = pd.DataFrame(predictions.float().numpy(), columns=classes)
        result.insert(0, ROW_ID, frame[ROW_ID].to_numpy())
        return result
