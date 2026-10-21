class AppError(Exception):
    """Base exception for application errors."""


class ConfigurationError(AppError):
    """Raised when the application cannot be configured."""
