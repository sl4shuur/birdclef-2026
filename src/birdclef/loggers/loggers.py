import logging
from collections.abc import Callable
from logging.config import dictConfig
from typing import Any

from birdclef.config import LoggingConfig

from .logging_formatters import ColoredFormatter, ContextualColorFormatter

FORMATTERS: dict[str, type[logging.Formatter]] = {
    ColoredFormatter.__name__: ColoredFormatter,
    ContextualColorFormatter.__name__: ContextualColorFormatter,
}


def build_logging_config(config: LoggingConfig) -> dict[str, Any]:
    console_formatter = FORMATTERS[config.formatter]

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "()": console_formatter,
                "full_color": config.full_color,
                "include_function": config.include_function,
                "date_format": config.date_format,
            },
            "eval_console": {
                "()": console_formatter,
                "full_color": config.full_color,
                "include_function": config.include_function,
                "date_format": config.date_format,
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": config.date_format,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": config.level,
                "formatter": "console",
                "stream": "ext://sys.stdout",
            },
            "eval_console": {
                "class": "logging.StreamHandler",
                "level": config.level,
                "formatter": "eval_console",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": config.level,
            "handlers": ["console"],
        },
        "loggers": {
            "eval": {
                "level": config.level,
                "handlers": ["eval_console"],
                "propagate": False,
            },
        },
    }


def apply_logging_config(config: LoggingConfig) -> None:
    logging.addLevelName(config.success_level, "SUCCESS")
    dictConfig(build_logging_config(config))


def get_logger[**P, TLogger: logging.LoggerAdapter](
    logger_factory: Callable[P, TLogger],
    *args: P.args,
    **kwargs: P.kwargs,
) -> TLogger:
    return logger_factory(*args, **kwargs)
