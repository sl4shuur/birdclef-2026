"""Feature-based MLP lifecycle; tensor preparation belongs to birdclef.data."""

import hashlib
import json
from pathlib import Path
from typing import Self

import lightning as L
import pandas as pd
from lightning.pytorch.utilities import rank_zero_only

from birdclef.competition import ROW_ID
from birdclef.data.datamodule import TrainingDataModule, prediction_dataloader
from birdclef.data.features import FeatureDataset, multilabel_dataset
from birdclef.experiment.contract import BirdClefModel, RunContext
from birdclef.models.baseline.config import BaselineExperimentConfig
from birdclef.models.baseline.network import BaselineModule
from birdclef.models.checkpoints import read_fit_state
from birdclef.models.trainer import fit_module, predict_module
from birdclef.submission.format import validate_row_ids


class Baseline(BirdClefModel):
    def __init__(self, config: BaselineExperimentConfig) -> None:
        self.config = config
        self._module: BaselineModule | None = None
        self._best: Path | None = None

    @classmethod
    def from_run(cls, output_dir: Path, *, config: BaselineExperimentConfig | None = None) -> Self:
        """Restore model settings and checkpoint selection; predict loads the saved weights."""
        saved = BaselineExperimentConfig.model_validate_json(
            (output_dir / "model-config.json").read_text(encoding="utf-8")
        )
        if config is not None:
            if config.model != saved.model or config.lr_scheduler != saved.lr_scheduler:
                raise ValueError("Inference must use the saved model settings.")
            saved.runtime = config.runtime
        model = cls(saved)
        model._module = BaselineModule(saved.model, saved.lr_scheduler)
        model._best = read_fit_state(output_dir).best_path(output_dir)
        return model

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, context: RunContext) -> None:
        config = self.config
        if train.empty or valid.empty:
            raise ValueError("Training and validation frames must be nonempty.")
        L.seed_everything(config.train.seed, workers=True)
        data = TrainingDataModule(
            multilabel_dataset(
                train,
                feature_columns=config.model.feature_columns,
                target_columns=config.model.target_columns,
            ),
            multilabel_dataset(
                valid,
                feature_columns=config.model.feature_columns,
                target_columns=config.model.target_columns,
            ),
            config=config.loader,
        )
        self._check_inputs(train, valid, context)
        context.tracker.log_hyperparams(config.safe_dump())
        self._module = BaselineModule(config.model, config.lr_scheduler)
        self._best = fit_module(
            self._module,
            data,
            train_config=config.train,
            loader_config=config.loader,
            runtime=config.runtime,
            run=context,
        )

    def predict(self, frame: pd.DataFrame, context: RunContext | None = None) -> pd.DataFrame:
        if self._module is None or self._best is None:
            raise RuntimeError("Call fit() or from_run() before predict().")
        ids = frame[ROW_ID]
        columns = self.config.model.target_columns
        if frame.empty:
            return pd.DataFrame(columns=[ROW_ID, *columns])
        validate_row_ids(frame)
        loader = prediction_dataloader(
            FeatureDataset(frame, self.config.model.feature_columns), config=self.config.loader
        )
        predictions = predict_module(
            self._module,
            loader,
            runtime=self.config.runtime,
            best=self._best,
        )
        result = pd.DataFrame(predictions.float().numpy(), columns=columns)
        result.insert(0, ROW_ID, ids.to_numpy())
        return result

    def _check_inputs(self, train: pd.DataFrame, valid: pd.DataFrame, context: RunContext) -> None:
        """Refuse changed data/settings; only epoch budget and checkpoint count may change."""
        output_dir = context.output_dir
        settings = self.config.safe_dump()
        keys = ("model", "loader", "lr_scheduler", "train")
        settings = {key: settings[key] for key in keys}
        for key in ("max_epochs", "checkpoint_top_k"):
            settings["train"].pop(key)
        columns = [*self.config.model.feature_columns, *self.config.model.target_columns]
        checksums = [
            hashlib.sha256(
                pd.util.hash_pandas_object(df[columns], index=True).values.tobytes()
            ).hexdigest()
            for df in (train, valid)
        ]
        signature = {"settings": settings, "data": checksums, "protocol": context.protocol}
        path = output_dir / "fit-inputs.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != signature:
                raise ValueError("Data or model settings changed; choose a new run directory.")
        elif (output_dir / "checkpoints").exists() or (output_dir / "fit.json").exists():
            raise ValueError("Existing checkpoints require fit-inputs.json to resume safely.")
        if rank_zero_only.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(signature, indent=2), encoding="utf-8")
            temporary.replace(path)
            (output_dir / "model-config.json").write_text(
                json.dumps(self.config.safe_dump(), indent=2),
                encoding="utf-8",
            )
