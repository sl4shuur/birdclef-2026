import logging
from pathlib import Path

from colorama import Fore, Style


def _hex_to_ansi(hex_color: str) -> str:
    """Convert hex color to ANSI escape sequence."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m"


# fmt: off
COLORS = {
    "DEBUG":    _hex_to_ansi("#3ACEFF"),
    "INFO":     _hex_to_ansi("#A1F7FF"),
    "SUCCESS":  _hex_to_ansi("#69FE69"),
    "WARNING":  _hex_to_ansi("#FDF32F"),
    "ERROR":    _hex_to_ansi("#F61C1C"),
    "CRITICAL": _hex_to_ansi("#FF6EFF"),
}

EVAL_THRESHOLDS = [
    ("GOOD", 0.69,  _hex_to_ansi("#69FE69")),
    ("OKAY", 0.4, _hex_to_ansi("#FDF32F")),
    ("POOR", 0.0,  _hex_to_ansi("#F61C1C")),
]
# fmt: on


PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../src/app


def _source_label(record: logging.LogRecord) -> str:
    path = Path(record.pathname).with_suffix("")

    if path.is_relative_to(PACKAGE_ROOT):
        parts = path.relative_to(PACKAGE_ROOT).parts
    elif "site-packages" in path.parts:
        parts = path.parts[path.parts.index("site-packages") + 1 :]

    else:
        parts = (path.name,)
    if len(parts) > 1 and parts[-1] == "__init__":
        parts = parts[:-1]
    return "/".join(parts)


def _get_eval_badge(score: float) -> tuple[str, str]:
    """
    Return (label, ansi_color) based on score thresholds.

    :param score: Float score in range [0, 1].
    :return: Tuple of badge label and ANSI color string.
    """
    for label, threshold, color in EVAL_THRESHOLDS:
        if score >= threshold:
            return label, color
    return "POOR", _hex_to_ansi("#F61C1C")


class ContextualColorFormatter(logging.Formatter):
    """
    Colored console formatter.

    Supports two layout modes:
    - inline (full_color=False): timestamp on one line, level + message on next
    - block  (full_color=True):  entire output wrapped in level color

    Supports two context modes (mutually exclusive, eval_mode takes priority):
    - include_function: show file path + line + function name
    - eval_mode:        show [GOOD]/[OKAY]/[POOR] badge from record.score
    """

    def __init__(
        self,
        full_color: bool = False,
        include_function: bool = False,
        eval_mode: bool = False,
        date_format: str = "%d-%m-%Y %H:%M:%S",
    ) -> None:
        """
        :param full_color: Wrap entire output in level color block.
        :param include_function: Show file path + line number + function name.
        :param eval_mode: Show score badge instead of function info (requires record.score).
        """
        super().__init__()
        self.full_color = full_color
        self.include_function = include_function
        self.eval_mode = eval_mode
        self.date_format = date_format

    def _build_context(self, record: logging.LogRecord, log_color: str) -> str | None:
        """
        Build context string: either eval badge or function info.

        :param record: The log record.
        :param log_color: ANSI color for the current log level.
        :return: Formatted context string or None.
        """
        if self.eval_mode:
            score = getattr(record, "score", None)
            if score is not None:
                label, badge_color = _get_eval_badge(float(score))
                if self.full_color:
                    return f"{Style.RESET_ALL}{badge_color}[{label}]{Style.RESET_ALL}{log_color}"
                # Do NOT reset after badge — message color will be applied separately
                return f"{badge_color}[{label}]"

        if self.include_function:
            file_info = f"{_source_label(record)}:{record.lineno} -> {record.funcName}"
            if self.full_color:
                return f"[{file_info}]"
            return f"{_hex_to_ansi('#76ADF4')}[{file_info}]{Style.RESET_ALL}"

        return None

    def format(self, record: logging.LogRecord) -> str:
        log_color = COLORS.get(record.levelname, Fore.WHITE)
        timestamp = self.formatTime(record, self.date_format)

        score = getattr(record, "score", None)

        # Determine message color based on badge if available, otherwise use level color
        has_score = False
        if self.eval_mode and score is not None:
            has_score = True
            _, message_color = _get_eval_badge(float(score))
        else:
            message_color = log_color

        # Build context AFTER message_color is determined
        context = self._build_context(record, log_color)

        if self.full_color:
            parts = [f"[{timestamp}]"]
            if context:
                parts.append(context)
            header = " ".join(parts)
            if has_score:
                block = f"\n{header}{Style.RESET_ALL}\n{message_color}{record.getMessage()}"
                return f"{message_color}{block}{Style.RESET_ALL}"
            block = f"\n{header}\n{record.levelname}: {record.getMessage()}"
            return f"{message_color}{block}{Style.RESET_ALL}"

        # Inline mode
        colored_time = f"{message_color}{timestamp}{Style.RESET_ALL}"
        parts = [f"[{colored_time}]"]
        if context:
            parts.append(context)
        header = " ".join(parts)

        if has_score:
            return f"{header} {message_color}{record.getMessage()}{Style.RESET_ALL}"

        colored_level = f"{message_color}{record.levelname}{Style.RESET_ALL}"
        return f"{header}\n{colored_level}: {message_color}{record.getMessage()}{Style.RESET_ALL}"


class ColoredFormatter(logging.Formatter):
    """
    Single-line aligned formatter.

    Example:
    15-04-2026 21:34:52  INFO  [services/pipeline:89]  message
    """

    def __init__(
        self,
        *,
        full_color: bool = True,
        include_function: bool = True,
        level_width: int = 8,
        date_format: str = "%d-%m-%Y %H:%M:%S",
    ) -> None:
        super().__init__()
        self.full_color = full_color
        self.include_function = include_function
        self.level_width = level_width
        self.date_format = date_format

    @staticmethod
    def _colorize(text: str, enabled: bool, color: str) -> str:
        if not enabled:
            return text
        return f"{color}{text}{Style.RESET_ALL}"

    def _format_message(self, record: logging.LogRecord) -> str:
        message = record.getMessage()

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{record.exc_text}"

        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        return message

    def _format_eval_record(self, record: logging.LogRecord, score: float) -> str:
        label, eval_color = _get_eval_badge(float(score))

        timestamp = self.formatTime(record, self.date_format)
        message = self._format_message(record)
        row = "  ".join([timestamp, label, message])

        return self._colorize(row, self.full_color, eval_color)

    def format(self, record: logging.LogRecord) -> str:
        score: float | None = getattr(record, "score", None)
        if score is not None:
            return self._format_eval_record(record, score)

        log_color = COLORS.get(record.levelname, Fore.WHITE)

        timestamp = self.formatTime(record, self.date_format)
        if self.include_function:
            level = f"{record.levelname:<{self.level_width}}"
        else:
            level = record.levelname
        message = self._format_message(record)

        timestamp = self._colorize(timestamp, self.full_color, log_color)
        level = self._colorize(level, self.full_color, log_color)

        parts = [timestamp, level]
        if self.include_function:
            location = f"\033[4m[{_source_label(record)}:{record.lineno}]\033[24m"
            location = self._colorize(location, self.full_color, _hex_to_ansi("#76ADF4"))
            parts.append(location)
        parts.append(message)

        return "  ".join(parts)


class EvalFileFormatter(logging.Formatter):
    def __init__(self, date_format: str = "%d-%m-%Y %H:%M:%S") -> None:
        super().__init__()
        self.date_format = date_format

    def format(self, record: logging.LogRecord) -> str:
        score: float | None = getattr(record, "score", None)
        level = record.levelname
        if score is not None:
            level, _ = _get_eval_badge(float(score))

        timestamp = self.formatTime(record, self.date_format)
        return f"{timestamp} [{level}] {record.getMessage()}"
