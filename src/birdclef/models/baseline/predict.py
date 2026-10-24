"""Species probabilities from a saved feature-based multilabel MLP."""

from pathlib import Path

import rich_click as click

from birdclef.cli_common import CONTEXT_SETTINGS, load_config
from birdclef.models.baseline.config import BaselineExperimentConfig


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--run", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--input", "input_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--accelerator", type=click.Choice(["auto", "cpu", "gpu"]), default="auto")
def predict(run: Path, input_path: Path, output: Path, accelerator: str) -> None:
    """Write row_id and species probabilities; validate with submission.format."""
    config = load_config(run / "model-config.json", BaselineExperimentConfig)
    config.runtime = config.runtime.model_copy(update={"accelerator": accelerator, "devices": 1})

    import pandas as pd

    from birdclef.models.baseline.model import Baseline

    try:
        predictions = Baseline.from_run(run, config=config).predict(pd.read_csv(input_path))
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output, index=False)
    except (OSError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Saved predictions: {output}")


if __name__ == "__main__":
    predict()
