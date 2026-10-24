"""The competition-specific part: swap model code without rewriting lifecycle."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from lightning.pytorch.loggers import Logger

from birdclef.loggers import CustomLogger


@dataclass(frozen=True)
class RunContext:
    output_dir: Path
    tracker: Logger
    logger: CustomLogger
    # Names the validation subset; scores of different protocols are never compared.
    protocol: str


class BirdClefModel(ABC):
    @abstractmethod
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, context: RunContext) -> None:
        """Train on the supplied split; save reusable artifacts inside context.output_dir."""

    @abstractmethod
    def predict(self, frame: pd.DataFrame, context: RunContext | None = None) -> pd.DataFrame:
        """Return row_id and species probabilities for each input 5-second window.

        Input rows have unique row_id values; output preserves their order.
        Species columns follow the sample submission. Models own audio loading
        and preprocessing; callers own the window manifest and submission checks.
        """
