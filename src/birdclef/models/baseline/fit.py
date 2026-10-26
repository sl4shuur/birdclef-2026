"""Train the numeric example on already separated train/validation CSV files.

Replace CSV loading with the competition's loading and split logic.
Notebook callers can call Baseline.fit directly with their own frames and RunContext.
"""

import os
import sys
from pathlib import Path

import rich_click as click

from birdclef.cli_common import CONTEXT_SETTINGS, fit_options, load_config
from birdclef.models import BaselineExperimentConfig


@click.command(context_settings=CONTEXT_SETTINGS)
@fit_options
@click.option("--train", "train_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--valid", "valid_path", type=click.Path(path_type=Path, exists=True), required=True)
def fit(config_path: Path, name: str, dataset_id: str, train_path: Path, valid_path: Path) -> None:
    """Train/resume the example; use a new name when data, model or split changes."""
    config = load_config(config_path, BaselineExperimentConfig)
    if not name or name in {".", ".."} or Path(name).name != name:
        raise click.ClickException("Run name must be a single folder name.")
    if sys.platform == "win32":
        os.environ.setdefault("USE_LIBUV", "0")

    import pandas as pd
    from lightning.pytorch.utilities import rank_zero_only

    from birdclef.experiment import RunContext
    from birdclef.loggers import CustomLogger, apply_logging_config, wandb_run
    from birdclef.models import Baseline

    apply_logging_config(config.logging)
    logger = CustomLogger(
        f"baseline.rank{getattr(rank_zero_only, 'rank', 0)}", config=config.logging
    )
    output_dir = config.data.output_dir / name
    try:
        train, valid = pd.read_csv(train_path), pd.read_csv(valid_path)
        with wandb_run(config.tracking, name=name, save_dir=output_dir) as tracker:
            context = RunContext(output_dir, tracker, logger, dataset_id)
            Baseline(config).fit(train, valid, context)
    except (OSError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error
    if getattr(rank_zero_only, "rank", 0) == 0:
        click.echo(f"Saved training run: {output_dir}")


if __name__ == "__main__":
    fit()
