"""Train the acoustic baseline from the CLI or a notebook's %run."""

import json
import os
import sys
from pathlib import Path

import rich_click as click

from birdclef.cli_common import CONTEXT_SETTINGS, load_config
from birdclef.models import AudioExperimentConfig


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--name", default="audio-cnn", show_default=True, help="Prefix for the config-derived run name."
)
@click.option(
    "--prepared-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Prepared artifacts; defaults to <data.output_dir>/data_prep.",
)
def fit(config_path: Path, name: str, prepared_dir: Path | None) -> None:
    """Train/resume using fixed folds and W&B."""
    config = load_config(config_path, AudioExperimentConfig)
    if sys.platform == "win32":
        os.environ.setdefault("USE_LIBUV", "0")

    import pandas as pd
    import wandb

    from birdclef.data import load_audio_split
    from birdclef.experiment import RunContext
    from birdclef.loggers import CustomLogger, apply_logging_config, wandb_run
    from birdclef.models import AudioCNN, read_fit_state

    apply_logging_config(config.logging)
    logger = CustomLogger("audio-baseline", config=config.logging)
    try:
        prepared_dir = prepared_dir or config.data.output_dir / "data_prep"
        classes = json.loads((prepared_dir / "classes.json").read_text(encoding="utf-8"))
        model = AudioCNN(config, classes=classes)
        run_name = model.run_name(name)
        output_dir = config.data.output_dir / run_name
        train, valid = load_audio_split(
            prepared_dir,
            config.data.competition_dir,
            fold=config.fold,
            classes=classes,
        )
        if config.smoke:
            train = pd.concat(
                [
                    part.sample(n=min(32, len(part)), random_state=config.train.seed)
                    for _, part in train.groupby("source", sort=False)
                ],
                ignore_index=True,
            )
        protocol = f"file-gkf5/fold-{config.fold}" + ("/smoke" if config.smoke else "")
        logger.info(
            f"Run: {run_name}; training examples: {len(train)}; validation windows: {len(valid)}"
        )
        with wandb_run(config.tracking, name=run_name, save_dir=output_dir) as tracker:
            model.fit(train, valid, RunContext(output_dir, tracker, logger, protocol))
            metrics = json.loads(
                (output_dir / "validation-metrics.json").read_text(encoding="utf-8")
            )
            tracker.log_table(
                key="validation/classes",
                dataframe=pd.read_csv(output_dir / "validation-class-metrics.csv"),
            )
            artifact = wandb.Artifact(
                f"model-{tracker.experiment.id}", type="model", metadata=metrics
            )
            checkpoint = read_fit_state(output_dir).best_path(output_dir)
            files = [
                checkpoint,
                output_dir / "model-config.json",
                output_dir / "classes.json",
                output_dir / "fit.json",
                output_dir / "fit-inputs.json",
                output_dir / "validation-predictions.csv",
                output_dir / "validation-class-metrics.csv",
                output_dir / "validation-metrics.json",
            ]
            for path in files:
                artifact.add_file(str(path), name=path.relative_to(output_dir).as_posix())
            tracker.experiment.log_artifact(artifact, aliases=["best"])
    except (OSError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Saved training run: {output_dir}")


if __name__ == "__main__":
    fit()
