# Evaluation

`metrics.score(solution, submission, row_id_column_name="row_id")` is a local adaptation of <https://www.kaggle.com/code/metric/birdclef-roc-auc> without Kaggle utilities.

It drops the ID, checks numeric prediction types, selects classes with positive target sums, and calls scikit-learn `roc_auc_score(..., average="macro")`. Inputs are not mutated. Rows must already be aligned; use `submission.build.build_submission` before scoring if needed. Undefined-AUC behavior follows scikit-learn. Compute on the full validation set.
