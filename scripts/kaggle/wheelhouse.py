"""Ship the project and missing dependencies to offline Kaggle notebooks."""

import rich_click as click

from scripts.kaggle.dataset import STAGE, download_wheels, upload, write_dataset_files
from scripts.kaggle.image import (
    FREEZE,
    missing_from_image,
    read_pins,
    shared_with_image,
    write_pins,
)
from scripts.kaggle.kernel import pip_freeze_on_kaggle
from scripts.wheelhouse_common import CONTEXT_SETTINGS, build_wheel, run


def _pin() -> None:
    click.echo("Resolving packages shared with the Kaggle image...")
    write_pins(shared_with_image())
    click.echo("Pins updated. Refreshing uv.lock...")
    run("uv", "lock")
    click.secho("Pins and lockfile updated.", fg="green")


@click.group(
    context_settings=CONTEXT_SETTINGS,
    epilog="Tip: use -h as a shortcut for --help.",
)
def main() -> None:
    """Ship the project package and the packages the Kaggle image lacks."""


@main.command()
def snapshot() -> None:
    """Record the packages of the Kaggle image, then pin."""
    click.secho(
        "Collecting pip freeze on Kaggle; the remote notebook may take several minutes...",
        fg="yellow",
    )
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(pip_freeze_on_kaggle(), encoding="utf-8")
    click.secho(f"Image snapshot saved to {FREEZE}", fg="green")
    _pin()


@main.command()
def pin() -> None:
    """Pin the packages we share with the image in pyproject.toml, and re-lock."""
    _pin()


@main.command()
@click.option("-m", "--message", help="Dataset version notes. Default: the last commit subject.")
@click.option("--dry-run", is_flag=True, help="Stage the wheels without uploading.")
@click.option(
    "--create",
    is_flag=True,
    help="Create the first Dataset version instead of updating an existing one.",
)
def publish(message: str | None, dry_run: bool, create: bool) -> None:
    """Upload birdclef and the packages the image lacks as a new dataset version."""
    click.secho(
        "Checking pins against the Kaggle image (resolving shared dependencies)...", fg="yellow"
    )
    if read_pins() != shared_with_image():
        raise click.ClickException(
            "The pins are stale. Run: uv run python -m scripts.kaggle.wheelhouse pin"
        )
    click.secho("Pins match the image.", fg="green")

    click.echo(f"Building the birdclef wheel in a fresh staging directory: {STAGE}...")
    version = build_wheel(STAGE)
    click.secho(f"Built birdclef=={version}.", fg="green")

    click.echo("Resolving dependencies missing from the Kaggle image...")
    missing = missing_from_image()
    click.echo(f"Resolved {len(missing)} packages to ship alongside birdclef.")
    download_wheels(missing)

    click.echo("Writing requirements.txt and dataset metadata...")
    click.echo(write_dataset_files(version, missing), nl=False)
    click.echo(f"Staged {len(list(STAGE.glob('*.whl')))} wheels in {STAGE}")
    if dry_run:
        click.secho("Dry run complete. No dataset version was uploaded.", fg="green")
        return

    notes = message or run("git", "log", "-1", "--format=%s").strip()
    click.echo(
        "Creating wheelhouse Dataset..."
        if create
        else "Uploading new wheelhouse Dataset version..."
    )
    upload(notes, create=create)
    click.secho("Wheelhouse published.", fg="green")


if __name__ == "__main__":
    main()
