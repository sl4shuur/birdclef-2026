"""Platform detection, startup defaults and notebook secret access."""

import os
import sys
from pathlib import Path
from typing import Literal

type Environment = Literal["kaggle", "colab", "local"]

_DEFAULTS: dict[Environment, dict[str, str]] = {
    # Adjust the input and Drive paths for your competition; mount Drive in Colab first.
    "kaggle": {
        "BCLEF_DATA__COMPETITION_DIR": "/kaggle/input/competitions/birdclef-2026",
        "BCLEF_DATA__OUTPUT_DIR": "/kaggle/working",
        "BCLEF_DATA__CACHE_DIR": "/kaggle/working/cache",
        "BCLEF_LOGGING__DIRECTORY": "/kaggle/working/logs",
        "BCLEF_LOADER__PIN_MEMORY": "true",
    },
    "colab": {
        "BCLEF_DATA__COMPETITION_DIR": "/content/drive/Shareddrives/BirdCLEF2026/data/competition",
        "BCLEF_DATA__OUTPUT_DIR": "/content/drive/Shareddrives/BirdCLEF2026/outputs",
        "BCLEF_DATA__CACHE_DIR": "/content/drive/Shareddrives/BirdCLEF2026/data/cache",
        "BCLEF_LOGGING__DIRECTORY": "/content/drive/Shareddrives/BirdCLEF2026/outputs/logs",
        "BCLEF_LOADER__PIN_MEMORY": "true",
    },
    "local": {},
}


def detect_environment(verbose: bool = False) -> Environment:
    env: Environment = "local"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists():
        env = "kaggle"
    elif "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        env = "colab"
    if verbose:
        print(f"Detected environment: {env}")
    return env


def apply_environment_defaults(environment: Environment | None = None) -> None:
    """Call before the first get_config(); existing environment variables take precedence."""
    if environment is None:
        environment = detect_environment(verbose=True)
    for name, value in _DEFAULTS[environment].items():
        os.environ.setdefault(name, value)


def get_secret(name: str, *, environment: Environment | None = None) -> str | None:
    """Read notebook secrets, or a process environment variable locally."""
    if environment is None:
        environment = detect_environment()
    match environment:
        case "kaggle":
            from kaggle_secrets import UserSecretsClient  # type: ignore

            return UserSecretsClient().get_secret(name)
        case "colab":
            from google.colab import userdata  # type: ignore

            return userdata.get(name)
        case "local":
            return os.environ.get(name)
