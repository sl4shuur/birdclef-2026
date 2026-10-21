from pathlib import Path

from birdclef.config import PROJECT_ROOT
from scripts.wheelhouse_common import run


def kaggle(*args: str, cwd: Path = PROJECT_ROOT) -> str:
    return run("uv", "tool", "run", "kaggle", *args, cwd=cwd)
