"""Submission format checks; no alignment, clipping or filling."""

import numpy as np
import pandas as pd

from birdclef.competition import ROW_ID, class_columns


def validate_row_ids(frame: pd.DataFrame) -> None:
    ids = frame[ROW_ID]
    if (
        frame.empty
        or not ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    ):
        raise ValueError("row_id must contain nonempty strings and at least one row.")
    if ids.duplicated().any():
        raise ValueError("Duplicate row_id values.")


def validate_submission(submission: pd.DataFrame, template: pd.DataFrame) -> None:
    """Require exact template column/row order and finite probabilities in [0, 1].

    The template must contain the actual expected IDs. On hidden inference, build
    it from soundscape_windows rather than copying the public sample's rows.
    """
    classes = class_columns(template)
    if submission.columns.tolist() != template.columns.tolist():
        raise ValueError("Submission columns or their order differ from the template.")
    validate_row_ids(template)
    validate_row_ids(submission)
    if submission[ROW_ID].tolist() != template[ROW_ID].tolist():
        raise ValueError("Submission row IDs or their order differ from the template.")
    if any(
        not pd.api.types.is_numeric_dtype(submission[c])
        or pd.api.types.is_complex_dtype(submission[c])
        for c in classes
    ):
        raise ValueError("Predictions must be numeric probabilities.")
    values = submission[classes].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
