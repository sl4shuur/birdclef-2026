"""Small, explicit Perch head ablations using the existing cached-feature pipeline."""

import argparse
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger

from birdclef.config import get_config
from birdclef.data import load_audio_split
from birdclef.experiment import RunContext
from birdclef.loggers import CustomLogger, wandb_run
from birdclef.models import read_fit_state
from birdclef.models.encoders.perch import PerchEncoder
from birdclef.models.runs import model_class, read_model_config

ROOT = Path("outputs/perch/search-20261029")
EXPERIMENTS = {
    "lstm256": ("bilstm", {"recurrent_dim": 256}, {}),
    "lstm128-head512": ("bilstm", {"hidden_dim": 512}, {}),
    "lstm128-sc03": ("bilstm", {}, {"soundscape_probability": 0.03}),
    "gru256": ("bigru", {"recurrent_dim": 256, "head_lr": 5e-4}, {}),
    "lstm256-reg": ("bilstm", {"recurrent_dim": 256, "dropout": 0.5, "head_lr": 5e-4}, {}),
    "lstm256-uni": ("bilstm", {"recurrent_dim": 256, "bidirectional": False, "head_lr": 5e-4}, {}),
    "mlp1024": ("mlp", {"hidden_dim": 1024, "dropout": 0.5}, {}),
    "mlp512-drop50": ("mlp", {"dropout": 0.5}, {}),
    "linear": ("mlp", {"hidden_dim": None}, {}),
    "mlp512-reg": ("mlp", {"dropout": 0.4, "head_lr": 5e-4, "weight_decay": 0.03}, {}),
    "mlp-baseline": ("mlp", {}, {}),
    "lstm-baseline": ("bilstm", {}, {}),
}


def configuration(name, fold, seed, online):
    base, model_changes, loader_changes = EXPERIMENTS[name]
    config = read_model_config(Path("outputs/perch") / base / "model-config.json")
    config.model = type(config.model).model_validate({**config.model.model_dump(), **model_changes})
    config.fold = fold
    config.train.max_epochs = 30
    config.train.early_stopping_patience = 6
    config.train.seed = seed
    config.loader.seed = seed
    config.loader.num_workers = 0
    for key, value in loader_changes.items():
        setattr(config.loader, key, value)
    config.runtime.accelerator = "gpu"
    config.runtime.devices = 1
    config.runtime.precision = "32-true"
    config.tracking = get_config(verbose=False).tracking.model_copy(update={"offline": not online})
    config.smoke = False
    return config


def run_experiment(name, fold, seed, online):
    config = configuration(name, fold, seed, online)
    slug = f"{name}-f{fold}-s{seed}" + ("-online" if online else "")
    directory = ROOT / slug
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "validation-metrics.json"
    if (directory / "search-result.json").exists():
        return json.loads((directory / "search-result.json").read_text())
    prepared = config.data.output_dir / "data_prep"
    classes = json.loads((prepared / "classes.json").read_text())
    train, valid = load_audio_split(
        prepared, config.data.competition_dir, fold=fold, classes=classes
    )
    model = model_class(config.model_type)(config, classes=classes)
    started = perf_counter()
    with (directory / "run.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            logger = CustomLogger("perch-search", config=config.logging)
            protocol = f"file-gkf5/fold-{fold}"
            # A cache miss must fail, rather than silently running costly ONNX inference.
            with patch.object(
                PerchEncoder, "__call__", side_effect=RuntimeError("Search needs a complete cache.")
            ):
                if online:
                    with wandb_run(
                        config.tracking, name=f"perch/search/{slug}", save_dir=directory
                    ) as tracker:
                        model.fit(train, valid, RunContext(directory, tracker, logger, protocol))
                        tracker.experiment.summary["search_candidate"] = name
                        tracker.experiment.summary["search_seed"] = seed
                        tracker.experiment.summary["search_fold"] = fold
                        (directory / "wandb-url.txt").write_text(tracker.experiment.url)
                else:
                    tracker = CSVLogger(directory, name="training")
                    model.fit(train, valid, RunContext(directory, tracker, logger, protocol))
    state = read_fit_state(directory)
    result = {
        "candidate": name,
        "fold": fold,
        "seed": seed,
        "online": online,
        **json.loads(metrics_path.read_text()),
        "epochs": state.trained_epochs,
        "checkpoint": state.best_checkpoint,
        "seconds": round(perf_counter() - started, 2),
        "run": str(directory),
    }
    (directory / "search-result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=list(EXPERIMENTS))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        print(f"START {name} fold={args.fold} seed={args.seed} online={args.online}", flush=True)
        print(json.dumps(run_experiment(name, args.fold, args.seed, args.online)), flush=True)
        results = [json.loads(path.read_text()) for path in ROOT.glob("*/search-result.json")]
        pd.DataFrame(results).sort_values(["fold", "auc"], ascending=[True, False]).to_csv(
            ROOT / "results.csv", index=False
        )


if __name__ == "__main__":
    main()
