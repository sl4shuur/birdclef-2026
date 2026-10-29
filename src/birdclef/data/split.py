"""Reproducible recording-level holdout; windows from a file stay together."""

from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from birdclef.competition import ROW_ID
from birdclef.core.schemas import DataSplit, SplitManifest


def group_holdout(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str = "filename",
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> DataSplit:
    """Hold out a fraction of groups, rounded up, keeping both partitions nonempty.

    The fraction is of groups, not rows. Rare-class coverage is not guaranteed.
    Use a broader group such as site when measuring generalization to new sites.
    Persist the returned manifest with experiment artifacts.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    groups = frame[group_column]
    if not groups.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Split group keys must be nonempty strings.")
    keys = sorted(groups.unique())
    if len(keys) < 2:
        raise ValueError("At least two recording groups are required.")
    shuffled = np.random.default_rng(seed).permutation(keys)
    n_valid = min(len(keys) - 1, max(1, int(np.ceil(len(keys) * validation_fraction))))
    valid_keys = sorted(shuffled[:n_valid].tolist())
    train_keys = sorted(shuffled[n_valid:].tolist())
    is_valid = groups.isin(valid_keys)
    train, valid = frame.loc[~is_valid].copy(), frame.loc[is_valid].copy()
    identity_columns = (
        list(dict.fromkeys([group_column, ROW_ID])) if ROW_ID in frame else [group_column]
    )
    identity = frame[identity_columns]
    checksum = sha256(
        pd.util.hash_pandas_object(identity, index=True).to_numpy().tobytes()
    ).hexdigest()
    return DataSplit(
        train=train,
        val=valid,
        manifest=SplitManifest(
            dataset_id=dataset_id,
            group_column=group_column,
            seed=seed,
            train_keys=train_keys,
            val_keys=valid_keys,
            train_rows=len(train),
            val_rows=len(valid),
            row_keys_checksum=checksum,
        ),
    )


def group_folds(
    frame: pd.DataFrame, *, n_splits: int = 5, group_column: str = "filename"
) -> pd.Series:
    """Zero-based GroupKFold assignments, stable under input row permutations.

    No label stratification or coverage guarantee. Save assignments by row_id;
    never recompute splits after filtering windows for a training experiment.
    """
    groups = frame[group_column]
    if not groups.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Split group keys must be nonempty strings.")
    if n_splits < 2 or groups.nunique() < n_splits:
        raise ValueError("Need at least n_splits distinct groups and n_splits >= 2.")
    order = np.argsort(groups.to_numpy(), kind="stable")
    assignments = np.full(len(frame), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, valid) in enumerate(splitter.split(order, groups=groups.iloc[order])):
        assignments[order[valid]] = fold
    return pd.Series(assignments, index=frame.index, name="fold")
