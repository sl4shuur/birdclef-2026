"""Raw scalar predictions from a saved run; competition scoring/submission is added later."""

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
    """Write raw scores in input-row order. This file is not a formatted Kaggle submission."""
    config = load_config(run / "model-config.json", BaselineExperimentConfig)
    config.runtime = config.runtime.model_copy(update={"accelerator": accelerator, "devices": 1})

    import pandas as pd

    from birdclef.models.baseline.model import Baseline

    try:
        predictions = Baseline.from_run(run, config=config).predict(pd.read_csv(input_path))
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"score": [prediction.score for prediction in predictions]}).to_csv(
            output,
            index=False,
        )
    except (OSError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Saved predictions: {output}")


if __name__ == "__main__":
    predict()
