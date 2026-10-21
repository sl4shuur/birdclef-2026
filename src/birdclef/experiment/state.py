"""Optional run-state stubs; implement if you need resume support.

Choose the state fields for your competition, e.g. run ID, phase and checkpoint.
Check configuration and split compatibility before resuming an existing run.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lightning.pytorch.utilities import rank_zero_only

from birdclef.core.schemas import Phase, RunState
from birdclef.loggers import CustomLogger


def load_state(output_dir: Path) -> dict[str, object]:
    """Read saved progress, or return initial state for a new run."""
    return {}


# Only rank 0 writes the state; `rank_zero_only` skips the call on every other rank.
@rank_zero_only
def save_state(state: RunState, output_dir: Path) -> None:
    temporary = output_dir / "state.json.tmp"
    temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_dir / "state.json")


def set_phase(state: RunState, phase: Phase, output_dir: Path) -> None:
    state.phase = phase
    save_state(state, output_dir)


@contextmanager
def run_lifecycle(state: RunState, *, output_dir: Path, logger: CustomLogger) -> Iterator[None]:
    """Record how training ended; final inference has its own entry point."""
    try:
        yield
    except (KeyboardInterrupt, SystemExit):
        state.phase = "interrupted"
        logger.warning(
            f"Training run {output_dir.name} interrupted; rerun the same output directory."
        )
        raise
    except Exception:
        state.phase = "failed"
        logger.error(f"Training run {output_dir.name} failed; rerun the same output directory.")
        raise
    else:
        state.phase = "complete"
        logger.info(f"Training run {output_dir.name} complete; artifacts saved to {output_dir}")
    finally:
        save_state(state, output_dir)
