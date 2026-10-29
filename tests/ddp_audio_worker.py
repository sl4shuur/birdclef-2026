"""Exercise the full audio lifecycle under script DDP on CPU or Kaggle GPUs."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from lightning.pytorch.utilities import rank_zero_only

from birdclef.config import (
    DataConfig,
    LoaderConfig,
    LoggingConfig,
    RuntimeConfig,
    TrackingConfig,
    TrainConfig,
)
from birdclef.experiment import RunContext
from birdclef.loggers import CustomLogger, wandb_run
from birdclef.models import AudioCNN, AudioCNNModule, AudioExperimentConfig, AudioModelConfig
from birdclef.models.audio_cnn import model as audio_model

ROOT = Path(sys.argv[1]).resolve()
ACCELERATOR = "gpu" if sys.argv[2] == "gpu" else "cpu"


class AuditedModule(AudioCNNModule):
    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        assert all(not parameter.requires_grad for parameter in self.encoder[0].parameters())
        assert all(
            parameter.requires_grad == (self.current_epoch >= 1)
            for parameter in self.encoder[-1].parameters()
        )
        self.before = next(self.encoder[-1].parameters()).detach().clone()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        weight = next(self.encoder[-1].parameters()).detach()
        gathered = self.all_gather(weight)
        assert isinstance(gathered, torch.Tensor)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        changed = not torch.equal(weight, self.before)
        assert changed == (self.current_epoch >= 1)
        (ROOT / f"rank-{self.global_rank}-epoch-{self.current_epoch}.json").write_text(
            json.dumps({"epoch": self.current_epoch, "encoder_updated": changed})
        )


def main() -> None:
    torch.set_num_threads(1)
    if sys.platform == "win32":
        import os

        os.environ.setdefault("USE_LIBUV", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if rank_zero_only.rank == 0:
        ROOT.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for name, seconds in [
            *[(f"focal-{index}", 5) for index in range(6)],
            ("training", 10),
            ("validation", 25),
        ]:
            sf.write(
                ROOT / f"{name}.wav",
                rng.standard_normal(seconds * 32000).astype("float32") * 0.1,
                32000,
            )
    training = pd.DataFrame(
        [
            {
                "source": "focal",
                "filename": f"focal-{index}.wav",
                "start": 0,
                "end": 5,
                "row_id": f"focal-{index}",
                "bird": index % 2,
                "owl": 1 - index % 2,
            }
            for index in range(6)
        ]
        + [
            {
                "source": "soundscape",
                "filename": "training.wav",
                "start": index * 5,
                "end": (index + 1) * 5,
                "row_id": f"training-{index}",
                "bird": index % 2,
                "owl": 1 - index % 2,
            }
            for index in range(2)
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "source": "soundscape",
                "filename": "validation.wav",
                "start": index * 5,
                "end": (index + 1) * 5,
                "row_id": f"validation-{index}",
                "bird": 1 - index % 2,
                "owl": index % 2,
            }
            for index in range(5)
        ]
    )
    for frame in [training, validation]:
        frame["path"] = frame.filename.map(lambda name: str(ROOT / str(name)))
    config = AudioExperimentConfig(
        data=DataConfig(output_dir=ROOT),
        model=AudioModelConfig(
            pretrained=ACCELERATOR == "gpu", unfreeze_epoch=1, unfreeze_last_n=2
        ),
        runtime=RuntimeConfig(accelerator=ACCELERATOR, devices=2),
        loader=LoaderConfig(
            batch_size=2,
            val_batch_size=2,
            predict_batch_size=2,
            num_workers=1,
            pin_memory=ACCELERATOR == "gpu",
            seed=42,
        ),
        train=TrainConfig(
            max_epochs=2,
            early_stopping_patience=3,
            log_every_n_steps=1,
            checkpoint_metric="val/auc",
            checkpoint_mode="max",
        ),
        tracking=TrackingConfig(offline=True),
    )
    config.logging = LoggingConfig(directory=ROOT / "logs")
    audio_model.AudioCNNModule = AuditedModule
    model = AudioCNN(config, classes=["bird", "owl"])
    logger = CustomLogger("ddp-check", config=config.logging)
    with wandb_run(config.tracking, name="ddp-check", save_dir=ROOT / "run") as tracker:
        model.fit(training, validation, RunContext(ROOT / "run", tracker, logger, "ddp-check"))
    if rank_zero_only.rank != 0:
        return
    metrics = json.loads((ROOT / "run/validation-metrics.json").read_text())
    assert metrics["windows"] == 5
    predictions = pd.read_csv(ROOT / "run/validation-predictions.csv")
    assert predictions.row_id.tolist() == validation.row_id.tolist()
    state = json.loads((ROOT / "run/fit.json").read_text())
    checkpoint = torch.load(ROOT / "run" / state["best_checkpoint"], weights_only=False)
    best = next(
        value["best_model_score"].item()
        for value in checkpoint["callbacks"].values()
        if value.get("best_model_score") is not None
    )
    assert abs(best - metrics["auc"]) < 1e-6
    assert len(list(ROOT.glob("rank-*-epoch-*.json"))) == 4
    assert len(list((ROOT / "run/wandb").glob("offline-run-*"))) == 1
    (ROOT / "ddp-check.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "accelerator": ACCELERATOR,
                "world_size": 2,
                "epochs": state["trained_epochs"],
                "auc": metrics["auc"],
                "validation_windows": 5,
                "encoder_gradients_synchronized": True,
                "wandb_runs": 1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
