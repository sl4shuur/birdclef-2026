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


def shared_with_image() -> dict[str, str]:
    """The image's versions of the packages birdclef installs that the image also has."""
    image = parse_freeze(_read_freeze())
    constraints = "".join(
        f"{name}=={version}\n" for name, version in image.items() if version and name != "torch"
    )

    # torch brings its CUDA packages, whose versions on the image can clash. We never pin torch.
    drop_torch = 'torch ; sys_platform == "never"\n'

    project = resolve(str(PYPROJECT), constraints=constraints, overrides=drop_torch)
    return {name: version for name, version in project.items() if name in image and name != "torch"}


def missing_from_image() -> dict[str, str]:
    """Everything birdclef and the notebooks need that the image lacks."""
    image = parse_freeze(_read_freeze())
    # A requirement with a marker that never holds is dropped from the resolution.
    # That leaves out the image's packages, and torch's CUDA dependencies with them.
    if "torch" not in image:
        raise click.ClickException("The snapshot has no PyTorch. Take a GPU image snapshot again.")
    overrides = "".join(f'{name} ; sys_platform == "never"\n' for name in image)
    return resolve(str(PYPROJECT), overrides=overrides)


def read_pins() -> dict[str, str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    block = text[text.index(PINS_BEGIN) : text.index(PINS_END)]
    return dict(re.findall(r'"([^"=]+)==([^"]+)"', block))


def write_pins(pins: dict[str, str]) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index(PINS_BEGIN) + len(PINS_BEGIN)
    lines = "".join(f'    "{name}=={version}",\n' for name, version in sorted(pins.items()))
    PYPROJECT.write_text(
        text[:start] + lines + text[text.index(PINS_END) :], encoding="utf-8", newline="\n"
    )


def resolve(*sources: str, constraints: str = "", overrides: str = "") -> dict[str, str]:
    """What `uv pip compile` installs on Kaggle's Linux image for `sources`."""
    # --no-config keeps the old pins in pyproject.toml from conflicting with a new image.
    with tempfile.TemporaryDirectory() as tmp:
        options = []
        for flag, text in (("--constraints", constraints), ("--override", overrides)):
            if text:
                file = Path(tmp) / f"{flag[2:]}.txt"
                file.write_text(text, encoding="utf-8")
                options += [flag, str(file)]
        compiled = run(
            *("uv", "pip", "compile", *sources, *options),
            *(flag for extra in EXTRAS for flag in ("--extra", extra)),
            *("--python-version", image_python(), "--python-platform", "x86_64-manylinux_2_28"),
            *("--no-header", "--no-annotate", "--quiet", "--no-config", "--no-sources"),
        )
    return parse_freeze(compiled)


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
