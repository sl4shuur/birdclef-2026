"""Experiment entry-point stub; implement the workflow for your competition.

Keep model architecture, dataset details and metric definitions in their own modules.
"""

from pathlib import Path

from birdclef.config import Config
from birdclef.core.schemas import DataSplit
from birdclef.experiment.contract import BirdClefModel
from birdclef.loggers import CustomLogger


def run_experiment(
    model: BirdClefModel, split: DataSplit, *, config: Config, name: str, logger: CustomLogger
) -> Path:
    """Train and validate, then return the run's output directory.

    Create the run folder, prepare data and split, initialize the model and tracker,
    save inputs, train, evaluate and save results. Add state/resume handling if needed.
    """
    return Path()
