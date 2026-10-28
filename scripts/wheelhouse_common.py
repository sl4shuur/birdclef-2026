"""Shared local build helpers for notebook bundles."""

import shutil
import subprocess
import tomllib
from pathlib import Path

import rich_click as click

from birdclef.cli_common import CONTEXT_SETTINGS as CONTEXT_SETTINGS
from birdclef.config import PROJECT_ROOT

EXTRAS = ("notebooks", "audio", "deep-learning")


def run(*command: str, cwd: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode:
        raise click.ClickException(f"{' '.join(command)}\n{result.stdout}{result.stderr}")
    return result.stdout


def build_wheel(stage: Path) -> str:
    """Build into a fresh project build subfolder and return the package version."""
    if stage.resolve().parent != (PROJECT_ROOT / "build").resolve():
        raise click.ClickException(
            "The staging folder must stay inside the project's build folder."
        )
    shutil.rmtree(stage, ignore_errors=True)
    run("uv", "build", "--wheel", "--out-dir", str(stage), "--quiet")
    (stage / ".gitignore").unlink(missing_ok=True)
    return next(stage.glob("*.whl")).name.split("-")[1]


def project_requirement(version: str) -> str:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return f"{metadata['project']['name']}[{','.join(EXTRAS)}]=={version}"
