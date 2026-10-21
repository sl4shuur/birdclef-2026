"""Shared CLI presentation and options; model commands add their own decorators."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import rich_click as click

from birdclef.config import Config

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "rich_help_config": {
        "text_markup": "rich",
        "show_arguments": True,
        "group_arguments_options": True,
        "style_option": "bold cyan",
        "style_command": "bold cyan",
        "style_usage": "bold yellow",
        "style_errors_panel_border": "red",
        "style_errors_suggestion": "magenta italic",
        "errors_suggestion": "\nTry running the [bold cyan]'--help'[/] flag for more information.",
        "errors_epilogue": "To find out more, visit our documentation.",
    },
}


def fit_options[F: Callable[..., Any]](function: F) -> F:
    """Apply below @click.command, alongside any model-specific @click.option."""
    function = click.option(
        "--dataset-id", required=True, help="Dataset version or fingerprint recorded in the split."
    )(function)
    function = click.option("--name", default="baseline", show_default=True)(function)
    return click.option(
        "--config",
        "config_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        required=True,
        help="Experiment configuration JSON.",
    )(function)


def predict_options[F: Callable[..., Any]](function: F) -> F:
    function = click.option(
        "--accelerator",
        type=click.Choice(["auto", "cpu", "gpu"]),
        default="auto",
        show_default=True,
    )(function)
    function = click.option(
        "--output-dir",
        type=click.Path(path_type=Path, file_okay=False),
        help="Result folder; defaults to <run>/<split>.",
    )(function)
    function = click.option(
        "--competition-dir",
        type=click.Path(path_type=Path, exists=True, file_okay=False),
        help="Override the saved data location.",
    )(function)
    function = click.option("--split", type=click.Choice(["val", "test"]), required=True)(function)
    return click.option(
        "--run",
        type=click.Path(path_type=Path, exists=True, file_okay=False),
        required=True,
        help="Saved training run folder.",
    )(function)


def load_config[C: Config](path: Path, config_class: type[C]) -> C:
    """Read the model's configuration with a CLI-friendly error."""
    try:
        return config_class.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise click.ClickException(f"Could not load configuration from {path}: {error}") from error
