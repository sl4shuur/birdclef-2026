"""Build a Colab bundle; pip downloads dependencies when installing."""

import shutil

import rich_click as click

from birdclef.config import PROJECT_ROOT
from scripts.wheelhouse_common import CONTEXT_SETTINGS, build_wheel, project_requirement

STAGE = PROJECT_ROOT / "build" / "colab-wheelhouse"
ARCHIVE = PROJECT_ROOT / "build" / "kaggle-starter-colab.zip"


@click.group(context_settings=CONTEXT_SETTINGS)
def main() -> None:
    """Prepare the package for Colab training and experiments."""


@main.command()
def build() -> None:
    """Build a folder and ZIP with the birdclef wheel and installation requirements."""
    click.echo(f"Building the birdclef wheel in {STAGE}...")
    version = build_wheel(STAGE)
    requirements = project_requirement(version) + "\n"
    (STAGE / "requirements.txt").write_text(requirements, encoding="utf-8")
    shutil.make_archive(str(ARCHIVE.with_suffix("")), "zip", root_dir=STAGE)
    click.echo(requirements, nl=False)
    click.secho(f"Ready: {STAGE}\nZIP: {ARCHIVE}", fg="green")


if __name__ == "__main__":
    main()
