"""Lightning resolves accelerator, devices and DDP itself; only the precision needs a rule."""

import torch

from birdclef.config import RuntimeConfig


def resolve_runtime(config: RuntimeConfig) -> RuntimeConfig:
    """Return the settings with `precision="auto"` replaced by a concrete value."""
    if config.precision != "auto":
        return config
    precision = "32-true"
    if config.accelerator != "cpu" and torch.cuda.is_available():
        # Compute capability 8 and above has native BF16; a T4 (7.5) does not.
        majors = [torch.cuda.get_device_capability(i)[0] for i in range(torch.cuda.device_count())]
        precision = "bf16-mixed" if all(major >= 8 for major in majors) else "16-mixed"
    return config.model_copy(update={"precision": precision})
