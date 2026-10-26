"""Public package API; optional ML dependencies load when their exports are used."""

from importlib import import_module
from typing import TYPE_CHECKING

from .environment import apply_environment_defaults, detect_environment, get_secret

if TYPE_CHECKING:
    from .hardware import resolve_runtime

__all__ = ["apply_environment_defaults", "detect_environment", "get_secret", "resolve_runtime"]

_MODULES = {"resolve_runtime": "hardware"}


def __getattr__(name: str) -> object:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
