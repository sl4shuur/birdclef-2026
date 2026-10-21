import json
import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from birdclef.config import LoggingConfig

from .logging_formatters import EvalFileFormatter


def _attach_file_handler(
    logger: logging.Logger,
    name: str,
    formatter: logging.Formatter | None = None,
    *,
    config: LoggingConfig,
) -> None:
    log_path = config.directory / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path.resolve())
        for h in logger.handlers
    ):
        return
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(config.level)
    handler.setFormatter(
        formatter or logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt=config.date_format,
        )
    )
    logger.addHandler(handler)


class CustomLogger(logging.LoggerAdapter):
    def __init__(self, name: str = "app", *, config: LoggingConfig) -> None:
        underlying = logging.getLogger(name)
        super().__init__(underlying, extra={})
        _attach_file_handler(underlying, name, config=config)
        self._success_level = config.success_level

    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stacklevel", 2)
        self.logger.log(self._success_level, message, *args, **kwargs)


class EvalLogger(logging.LoggerAdapter):
    def __init__(self, config: LoggingConfig, name: str = "eval") -> None:
        logger_name = name if name == "eval" or name.startswith("eval.") else f"eval.{name}"
        underlying = logging.getLogger(logger_name)
        super().__init__(underlying, {})
        _attach_file_handler(
            underlying,
            logger_name,
            EvalFileFormatter(config.date_format),
            config=config,
        )

        self._json_path = config.directory / "evaluation.json"
        self._run_id = datetime.now().strftime(config.date_format)

    def process(
        self,
        msg: Any,
        kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        extra = dict(self.extra) if self.extra else {}
        record_extra = kwargs.get("extra")

        if isinstance(record_extra, Mapping):
            extra.update(record_extra)

        kwargs["extra"] = extra
        return msg, kwargs

    def metric(self, label: str, score: float) -> None:
        self.info(f"{label}: {score:.4f}", extra={"score": score}, stacklevel=2)

    def flush_json(
        self,
        metrics: dict[str, Any],
        extra: dict[str, Any] | None = None,
        false_positives: list[dict[str, Any]] | None = None,
    ) -> None:
        runs: list[dict[str, Any]] = []
        if self._json_path.exists():
            try:
                runs = json.loads(self._json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                runs = []

        entry: dict[str, Any] = {
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        if extra:
            entry["extra"] = extra
        if false_positives:
            entry["false_positives"] = false_positives

        runs.append(entry)
        self._json_path.write_text(
            json.dumps(runs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
