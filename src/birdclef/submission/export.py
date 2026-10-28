"""Pack saved runs for uploading as a Kaggle Dataset."""

import json
import shutil
import tempfile
from pathlib import Path

import rich_click as click

from birdclef.cli_common import CONTEXT_SETTINGS


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default="build/birdclef-models.zip",
    show_default=True,
)
def main(runs: tuple[Path, ...], output: Path) -> None:
    """ZIP config, classes, fit.json and the best checkpoint of each run."""
    from birdclef.models import AudioExperimentConfig, read_fit_state

    if output.suffix != ".zip":
        raise click.ClickException("Output must be a .zip file.")
    if len({run.name for run in runs}) != len(runs):
        raise click.ClickException("Run folder names must be unique inside the archive.")
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for run in runs:
                config = AudioExperimentConfig.model_validate_json(
                    (run / "model-config.json").read_text(encoding="utf-8")
                )
                checkpoint = read_fit_state(run).best_path(run)
                destination = root / run.name
                destination.mkdir()
                (destination / "model-config.json").write_text(
                    json.dumps(config.safe_dump(), indent=2), encoding="utf-8"
                )
                shutil.copyfile(run / "classes.json", destination / "classes.json")
                shutil.copyfile(run / "fit.json", destination / "fit.json")
                target = destination / checkpoint.relative_to(run)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checkpoint, target)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.make_archive(str(output.with_suffix("")), "zip", root)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Ready for Kaggle Dataset upload: {output}")


if __name__ == "__main__":
    main()
