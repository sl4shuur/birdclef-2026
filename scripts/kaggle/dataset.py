import json

import rich_click as click

from birdclef.config import PROJECT_ROOT
from scripts.kaggle.image import image_python
from scripts.kaggle.shell import kaggle
from scripts.wheelhouse_common import project_requirement, run

STAGE = PROJECT_ROOT / "build" / "kaggle-wheelhouse"
DATASET = "denysmykhailov/birdclef-2026-wheelhouse"


def download_wheels(packages: dict[str, str]) -> None:
    if not packages:
        return
    python = image_python()
    print(f"Downloading Linux wheels for Python {python}...")
    run(
        *("uv", "tool", "run", "pip", "download", *specs(packages), "--no-deps", "--quiet"),
        *("--only-binary=:all:", "--implementation", "cp", "--python-version", python),
        *("--abi", "cp" + python.replace(".", ""), "--dest", str(STAGE)),
        *("--platform", "manylinux_2_28_x86_64", "--platform", "manylinux_2_17_x86_64"),
        *("--platform", "manylinux2014_x86_64"),
    )
    print(f"Wheels for Python {python} are ready.")


def write_dataset_files(version: str, packages: dict[str, str]) -> str:
    """Write the offline requirements and Kaggle Dataset metadata."""
    requirements = "\n".join([project_requirement(version), *specs(packages)]) + "\n"
    (STAGE / "requirements.txt").write_text(requirements, encoding="utf-8")
    metadata = {
        "title": "BirdCLEF 2026",
        "id": DATASET,
        "licenses": [{"name": "other"}],
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return requirements


def upload(message: str, *, create: bool = False) -> None:
    # The CLI builds a file name from the path it gets, so run it inside the folder.
    if create:
        response = kaggle("datasets", "create", "-p", ".", "-r", "zip", "-t", cwd=STAGE)
    else:
        response = kaggle(
            "datasets", "version", "-p", ".", "-m", message, "-r", "zip", "-t", cwd=STAGE
        )
    click.echo(response)
    # Kaggle can report an API error while returning a successful process exit code.
    if "creation error:" in response.lower():
        raise click.ClickException("Kaggle rejected the upload; see its response above.")


def specs(packages: dict[str, str]) -> list[str]:
    return [f"{name}=={version}" for name, version in packages.items()]
