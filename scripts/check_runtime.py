"""Optional PyTorch/Lightning hardware check. Install with --extra deep-learning."""

from birdclef.config import Config
from birdclef.runtime.hardware import resolve_runtime


def main() -> None:
    import lightning as L

    cfg = resolve_runtime(Config().runtime)
    trainer = L.Trainer(**cfg.model_dump(), logger=False, enable_checkpointing=False)
    print(
        f"{type(trainer.accelerator).__name__}; "
        f"{trainer.num_devices} device(s); {trainer.precision}"
    )


if __name__ == "__main__":
    main()
