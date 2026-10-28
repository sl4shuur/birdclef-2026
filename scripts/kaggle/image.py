import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

import rich_click as click

from birdclef.config import PROJECT_ROOT
from scripts.wheelhouse_common import EXTRAS, run

# fmt: off
NOTEBOOKS = PROJECT_ROOT / "notebooks" / "kaggle"
FREEZE    = NOTEBOOKS / "image-freeze.txt"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

PINS_BEGIN = "    # BEGIN kaggle-image-pins\n"
PINS_END   = "    # END kaggle-image-pins"
# fmt: on


def _read_freeze() -> str:
    if not FREEZE.exists():
        raise click.ClickException(
            f"{FREEZE.name} is missing. Run: uv run python -m scripts.kaggle.wheelhouse snapshot"
        )
    return FREEZE.read_text(encoding="utf-8")


def image_python() -> str:
    """Python major.minor recorded by the snapshot, used for resolution and downloads."""
    match = re.search(r"^# Kaggle default image, Python (\d+\.\d+)\.", _read_freeze())
    if match is None:
        raise click.ClickException(
            "Snapshot has no Python version. "
            "Run: uv run python -m scripts.kaggle.wheelhouse snapshot"
        )
    return match[1]


def _resolve_image() -> tuple[dict[str, str], dict[str, str]]:
    """Prefer installed versions, upgrading incompatible packages and their dependencies."""
    image = parse_freeze(_read_freeze())
    if "torch" not in image:
        raise click.ClickException("The snapshot has no PyTorch. Take a GPU image snapshot again.")
    preferences = "".join(
        f"{name}=={version}\n" for name, version in image.items() if version and name != "torch"
    )
    # Keep the image's PyTorch and CUDA stack out of the wheelhouse.
    project = resolve(
        str(PYPROJECT), preferences=preferences, overrides='torch ; sys_platform == "never"\n'
    )
    return image, project


def shared_with_image() -> dict[str, str]:
    """Project dependencies that can reuse the image's installed versions."""
    image, project = _resolve_image()
    return {name: version for name, version in project.items() if image.get(name) == version}


def missing_from_image() -> dict[str, str]:
    """Project dependencies that must be installed or upgraded on the image."""
    image, project = _resolve_image()
    return {name: version for name, version in project.items() if image.get(name) != version}


def write_pins(pins: dict[str, str]) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index(PINS_BEGIN) + len(PINS_BEGIN)
    lines = "".join(f'    "{name}=={version}",\n' for name, version in sorted(pins.items()))
    PYPROJECT.write_text(
        text[:start] + lines + text[text.index(PINS_END) :], encoding="utf-8", newline="\n"
    )


def resolve(*sources: str, preferences: str = "", overrides: str = "") -> dict[str, str]:
    """What `uv pip compile` installs on Kaggle's Linux image for `sources`."""
    # --no-config keeps the old pins in pyproject.toml from conflicting with a new image.
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "requirements.txt"
        # uv treats existing output pins as preferences, rather than hard constraints.
        output.write_text(preferences, encoding="utf-8")
        options = ["--output-file", str(output)]
        if overrides:
            file = Path(tmp) / "overrides.txt"
            file.write_text(overrides, encoding="utf-8")
            options += ["--override", str(file)]
        run(
            *("uv", "pip", "compile", *sources, *options),
            *(flag for extra in EXTRAS for flag in ("--extra", extra)),
            *("--python-version", image_python(), "--python-platform", "x86_64-manylinux_2_28"),
            *("--no-header", "--no-annotate", "--quiet", "--no-config", "--no-sources"),
        )
        return parse_freeze(output.read_text(encoding="utf-8"))


def parse_freeze(text: str) -> dict[str, str]:
    """Package name to version, from `name==1.0` lines and `name @ url-of-a-wheel` lines."""
    packages = {}
    for line in text.splitlines():
        if " @ " in line:
            name, _, url = line.partition(" @ ")
            filename = unquote(urlsplit(url).path).rsplit("/", 1)[-1]
            version = filename.split("-")[1] if filename.endswith(".whl") else ""
        elif "==" in line:
            name, _, version = line.partition("==")
        else:
            continue
        # Local versions such as `2.11.0+cpu` are dropped, PyPI does not have them.
        packages[re.sub(r"[-_.]+", "-", name.strip()).lower()] = re.split(r"\+|%2B|\s", version)[0]
    return packages
