import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

from birdclef.config import TrackingConfig


def _open(config: TrackingConfig, run_id: str | None, name: str, save_dir: Path) -> WandbLogger:
    tracker = WandbLogger(
        project=config.project,
        entity=config.entity,
        name=name,
        id=run_id,
        resume=None if config.offline else "allow",
        save_dir=save_dir,
        # Offline runs are synced later with `wandb sync`, which Kaggle notebooks need.
        offline=config.offline,
    )
    tracker.experiment.define_metric("val/auc", summary="max")
    tracker.experiment.define_metric("val/loss", summary="min")
    return tracker


@contextmanager
def wandb_run(
    config: TrackingConfig, *, name: str, save_dir: Path, run_id: str | None = None
) -> Iterator[WandbLogger]:
    """Open the W&B run for training. The run is closed on exit, even after an error.

    Without an explicit run_id, persist the identity in wandb-run-id.txt in save_dir.
    Online runs can resume logging; offline launches create local sessions for later sync.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None and getattr(rank_zero_only, "rank", 0) == 0:
        path = save_dir / "wandb-run-id.txt"
        if path.exists():
            run_id = path.read_text(encoding="utf-8").strip()
            if not run_id:
                raise ValueError(f"Empty W&B run ID: {path}")
        else:
            run_id = uuid4().hex
            temporary = path.with_suffix(".txt.tmp")
            temporary.write_text(run_id, encoding="utf-8")
            temporary.replace(path)
    if wandb.run is not None:
        # A cell interrupted earlier can leave its run open, and one id cannot be opened twice.
        wandb.run.finish()
    exit_code = 0
    try:
        api_key = os.environ.get("WANDB_API_KEY") or (
            config.api_key.get_secret_value() if config.api_key is not None else None
        )
        if getattr(rank_zero_only, "rank", 0) == 0 and api_key and not config.offline:
            wandb.login(key=api_key)
        tracker = _open(config, run_id, name, save_dir)
        yield tracker
    except BaseException:
        exit_code = 1
        raise
    finally:
        # `tracker.experiment` would start a run when `init` failed and hide the real error.
        if wandb.run is not None:
            wandb.run.finish(exit_code=exit_code)
