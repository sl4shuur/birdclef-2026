"""Public package API; optional ML dependencies load when their exports are used."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio import log_mel

__all__ = ["log_mel"]

_MODULES = {"log_mel": "audio"}


def __getattr__(name: str) -> object:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
