"""Build a training bundle or publish an offline Kaggle wheelhouse."""

import json
import shutil
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

import rich_click as click

from birdclef.config import PROJECT_ROOT, get_config
from scripts.kaggle.dataset import STAGE, download_wheels, upload, write_dataset_files
from scripts.kaggle.image import (
    FREEZE,
    missing_from_image,
    shared_with_image,
    write_pins,
)
from scripts.kaggle.kernel import pip_freeze_on_kaggle
from scripts.wheelhouse_common import CONTEXT_SETTINGS, build_wheel, project_requirement, run


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
@click.option(
    "--prepared-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Prepared audio artifacts; defaults to <data.output_dir>/data_prep.",
)
def build(prepared_dir: Path | None) -> str:
    """Bundle the current package, prepared data and ImageNet weights for an online GPU run."""
    import torch
    from torchvision.models import EfficientNet_B0_Weights

    prepared_dir = prepared_dir or get_config(verbose=False).data.output_dir / "data_prep"
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts_sha256"].items():
        if sha256((prepared_dir / name).read_bytes()).hexdigest() != expected:
            raise click.ClickException(f"Preparation artifact checksum mismatch: {name}")

    stage = STAGE
    version = build_wheel(stage)
    (stage / "requirements.txt").write_text(project_requirement(version) + "\n", encoding="utf-8")
    prepared = stage / "data_prep"
    prepared.mkdir()
    for name in ["manifest.json", *manifest["artifacts_sha256"]]:
        shutil.copyfile(prepared_dir / name, prepared / name)

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    weights.get_state_dict(progress=True, check_hash=True)
    filename = Path(urlsplit(weights.url).path).name
    cache = stage / "torch" / "hub" / "checkpoints"
    cache.mkdir(parents=True)
    shutil.copyfile(Path(torch.hub.get_dir()) / "checkpoints" / filename, cache / filename)
    archive = shutil.make_archive(str(PROJECT_ROOT / "build" / "birdclef-kaggle"), "zip", stage)
    click.secho(f"Ready for Kaggle Dataset upload: {archive}", fg="green")
    return version


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
@click.pass_context
def publish(context: click.Context, message: str | None, dry_run: bool, create: bool) -> None:
    """Publish the training bundle with missing or incompatible image dependencies."""
    click.echo("Resolving dependencies to install or upgrade on Kaggle...")
    missing = missing_from_image()
    click.secho(f"Resolved {len(missing)} packages to ship alongside birdclef.", fg="green")
    version = context.invoke(build, prepared_dir=None)
    download_wheels(missing)

    click.echo("Writing requirements.txt and dataset metadata...")
    click.echo(write_dataset_files(version, missing), nl=False)
    click.echo(f"Staged {len(list(STAGE.glob('*.whl')))} wheels in {STAGE}")
    if dry_run:
        click.secho("Dry run complete. No dataset version was uploaded.", fg="green")
        return

    notes = message or run("git", "log", "-1", "--format=%s").strip()
    click.secho(
        "Creating wheelhouse Dataset..."
        if create
        else "Uploading new wheelhouse Dataset version...",
        fg="yellow"
    )
    upload(notes, create=create)
    click.secho("Wheelhouse published.", fg="green")


if __name__ == "__main__":
    main()
