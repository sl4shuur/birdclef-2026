"""Download competition data using the existing Kaggle CLI wrapper."""

from pathlib import Path
from zipfile import BadZipFile, ZipFile

import rich_click as click

from birdclef.cli_common import CONTEXT_SETTINGS
from birdclef.competition import COMPETITION
from birdclef.config import get_config
from scripts.kaggle.shell import kaggle


def extract_archive(archive: Path, destination: Path) -> None:
    """Keep every extracted member inside the destination, including on Windows."""
    root = destination.resolve()
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive member escapes the data directory: {member.filename}")
        zipped.extractall(root)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--unzip/--no-unzip", default=True, show_default=True)
def download(output_dir: Path | None, unzip: bool) -> None:
    """Download BirdCLEF+ 2026 (accept rules and configure Kaggle credentials first)."""
    destination = output_dir or get_config(verbose=False).data.competition_dir
    destination.mkdir(parents=True, exist_ok=True)
    click.secho(f"Downloading {COMPETITION} to {destination}...", fg="yellow")
    kaggle("competitions", "download", "-c", COMPETITION, "-p", str(destination))
    if unzip:
        try:
            extract_archive(destination / f"{COMPETITION}.zip", destination)
        except (OSError, ValueError, BadZipFile) as error:
            raise click.ClickException(str(error)) from error
    click.secho(f"Data ready: {destination}", fg="green")


if __name__ == "__main__":
    download()
