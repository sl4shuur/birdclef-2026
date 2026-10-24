# Evaluation

`metrics.score(solution, submission, row_id_column_name="row_id")` is a local adaptation of <https://www.kaggle.com/code/metric/birdclef-roc-auc> without Kaggle utilities.

It drops the ID, checks numeric prediction types, selects classes with positive target sums, and calls scikit-learn `roc_auc_score(..., average="macro")`. Inputs are not mutated. Row IDs must be nonnull, unique and already aligned; mismatches raise ValueError. Use `submission.build.build_submission` before scoring if needed. Classes without negatives raise ValueError naming the classes, rather than producing NaN. Species columns must match and solution labels must be binary. Compute on the full validation set.
