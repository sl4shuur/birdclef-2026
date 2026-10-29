"""Pack saved runs for uploading as a Kaggle Dataset."""

import json
import shutil
import tempfile
from pathlib import Path

import rich_click as click

from birdclef.cli_common import CONTEXT_SETTINGS


def _discover_runs(roots: tuple[Path, ...]) -> dict[Path, Path]:
    from birdclef.models import read_fit_state

    candidates = sorted(
        {path.parent.resolve() for root in roots for path in root.rglob("model-config.json")}
    )
    found = {}
    for run in candidates:
        try:
            if not (run / "classes.json").is_file():
                raise FileNotFoundError(f"Missing {run / 'classes.json'}")
            found[run] = read_fit_state(run).inference_path(run)
        except (OSError, ValueError) as error:
            click.secho(f"Skipping {run}: {error}", fg="yellow")
    return found


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="A run folder or a parent folder to search recursively; repeat for several roots.",
)
@click.option(
    "--all", "select_all", is_flag=True, help="Export all discovered runs without prompting."
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default="build/birdclef-models.zip",
    show_default=True,
)
def main(runs: tuple[Path, ...], select_all: bool, output: Path) -> None:
    """Find saved models, select them with checkboxes, and pack an inference ZIP."""
    from birdclef.models import AudioExperimentConfig, read_fit_state
    from birdclef.models.classifier.encoders import encoder_class

    if output.suffix != ".zip":
        raise click.ClickException("Output must be a .zip file.")
    found = _discover_runs(runs)
    if not found:
        raise click.ClickException(
            "No exportable runs found: config, classes and a checkpoint are required."
        )
    if select_all:
        selected = list(found)
    else:
        import questionary

        selected = questionary.checkbox(
            "Select models to export",
            choices=[
                questionary.Choice(
                    title=f"{run}  [{checkpoint.relative_to(run)}]",
                    value=run,
                )
                for run, checkpoint in found.items()
            ],
            instruction="(arrows: move, space: select, a: all, Enter: export)",
        ).ask()
        if selected is None:
            raise click.Abort()
    if not selected:
        click.echo("Nothing selected; no archive written.")
        return
    if len({run.name for run in selected}) != len(selected):
        raise click.ClickException("Run folder names must be unique inside the archive.")
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for run in selected:
                config = AudioExperimentConfig.model_validate_json(
                    (run / "model-config.json").read_text(encoding="utf-8")
                )
                checkpoint = found[run]
                relative = checkpoint.relative_to(run)
                click.echo(f"Exporting {run.name}: {relative}")
                destination = root / run.name
                destination.mkdir()
                (destination / "model-config.json").write_text(
                    json.dumps(config.safe_dump(), indent=2), encoding="utf-8"
                )
                shutil.copyfile(run / "classes.json", destination / "classes.json")
                for name in encoder_class(config.model.encoder).artifacts:
                    shutil.copyfile(run / name, destination / name)
                state = read_fit_state(run).model_copy(
                    update={"best_checkpoint": relative.as_posix()}
                )
                (destination / "fit.json").write_text(
                    state.model_dump_json(indent=2), encoding="utf-8"
                )
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checkpoint, target)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.make_archive(str(output.with_suffix("")), "zip", root)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.secho(f"Ready for Kaggle Dataset upload: {output}", fg="green")


if __name__ == "__main__":
    main()
