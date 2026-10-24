"""Assemble species predictions in the expected row and class order."""

import pandas as pd

from birdclef.competition import ROW_ID, class_columns
from birdclef.submission.format import validate_row_ids, validate_submission


def build_submission(predictions: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """Align explicit IDs/classes to a template, rejecting missing or extra rows."""
    class_columns(template)
    if not predictions.columns.is_unique or set(predictions.columns) != set(template.columns):
        raise ValueError("Prediction columns must match row_id and all template species.")
    validate_row_ids(template)
    validate_row_ids(predictions)
    if set(predictions[ROW_ID]) != set(template[ROW_ID]):
        raise ValueError("Prediction IDs differ from the template (missing or extra rows).")
    submission = predictions.set_index(ROW_ID).loc[template[ROW_ID]].reset_index()
    submission = submission[template.columns].copy()
    validate_submission(submission, template)
    return submission
