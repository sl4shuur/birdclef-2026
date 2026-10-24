"""Local adaptation of https://www.kaggle.com/code/metric/birdclef-roc-auc."""

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score

from birdclef.competition import ROW_ID


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = ROW_ID
) -> float:
    """Macro ROC-AUC excluding classes with no positive labels.

    As in the organizer metric, rows must already be aligned. Unlike its notebook,
    this function does not mutate inputs and uses ValueError for input errors.
    Undefined AUC and other scoring errors follow scikit-learn's behavior.
    """
    solution = solution.drop(columns=row_id_column_name)
    submission = submission.drop(columns=row_id_column_name)

    if not is_numeric_dtype(submission.values):
        bad_dtypes = {
            column: submission[column].dtype
            for column in submission.columns
            if not is_numeric_dtype(submission[column])
        }
        raise ValueError(f"Invalid submission data types found: {bad_dtypes}")

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    if not scored_columns:
        raise ValueError("ROC-AUC requires at least one class with positive labels.")

    return float(
        roc_auc_score(
            solution[scored_columns].values,
            submission[scored_columns].values,
            average="macro",
        )
    )
