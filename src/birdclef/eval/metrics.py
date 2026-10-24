"""Local adaptation of https://www.kaggle.com/code/metric/birdclef-roc-auc."""

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score

from birdclef.competition import ROW_ID


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = ROW_ID
) -> float:
    """Macro ROC-AUC excluding classes with no positive labels.

    Rows must have matching, unique IDs in the same order. Classes without
    positives are skipped; classes without negatives raise ValueError because
    their AUC is undefined. Inputs are not mutated. Scores may be logits.
    """
    if not solution.columns.is_unique or not submission.columns.is_unique:
        raise ValueError("Solution and submission columns must be unique.")
    for frame in (solution, submission):
        ids = frame[row_id_column_name]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError("Solution and submission row IDs must be nonnull and unique.")
    if solution[row_id_column_name].tolist() != submission[row_id_column_name].tolist():
        raise ValueError("Solution and submission must have aligned row IDs.")
    solution = solution.drop(columns=row_id_column_name)
    submission = submission.drop(columns=row_id_column_name)
    if set(solution.columns) != set(submission.columns):
        raise ValueError("Solution and submission must have the same species columns.")
    if not solution.isin([0, 1]).to_numpy().all():
        raise ValueError("Solution labels must be binary and nonnull.")

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
    without_negatives = list(solution_sums[solution_sums == len(solution)].index)
    if without_negatives:
        raise ValueError(
            f"ROC-AUC is undefined for classes without negative labels: {without_negatives}"
        )

    return float(
        roc_auc_score(
            solution[scored_columns].values,
            submission[scored_columns].values,
            average="macro",
        )
    )
