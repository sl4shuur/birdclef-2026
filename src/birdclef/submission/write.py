"""Validate and write an already assembled submission."""

from pathlib import Path

import pandas as pd

from birdclef.submission.format import validate_submission


def write_submission(submission: pd.DataFrame, path: Path, template: pd.DataFrame) -> Path:
    """Validate, then write CSV without an index. Kaggle expects submission.csv."""
    validate_submission(submission, template)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path
