"""python -m birdclef.submission --run <saved run> [--run <another run>]."""

from pathlib import Path

import rich_click as click

from birdclef.cli_common import CONTEXT_SETTINGS
from birdclef.submission.predict import run_inference


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Saved run folder; repeat to average model probabilities.",
)
@click.option(
    "--weight",
    "weights",
    multiple=True,
    type=click.FloatRange(min=0, min_open=True),
    help="One weight per --run; defaults to equal weights.",
)
@click.option(
    "--competition-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default="submission.csv",
    show_default=True,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--num-workers", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--threads", type=click.IntRange(min=1), default=4, show_default=True)
@click.option(
    "--soundscapes",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Explicit audio folder for a local check; never writes sample placeholders.",
)
def main(
    runs: tuple[Path, ...],
    weights: tuple[float, ...],
    competition_dir: Path,
    output: Path,
    batch_size: int,
    num_workers: int,
    threads: int,
    soundscapes: Path | None,
) -> None:
    """Write an offline CPU submission using the best checkpoints of saved runs."""
    import torch

    torch.set_num_threads(threads)
    try:
        run_inference(
            runs,
            competition_dir,
            output,
            weights=weights or None,
            batch_size=batch_size,
            num_workers=num_workers,
            soundscapes=soundscapes,
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
