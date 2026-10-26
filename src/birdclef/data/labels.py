"""Multi-hot targets for focal recordings in the official train.csv."""

import ast

import numpy as np
import pandas as pd

from birdclef.competition import ROW_ID


def encode_labels(
    frame: pd.DataFrame, classes: list[str], *, include_secondary: bool = True
) -> pd.DataFrame:
    """Encode primary/secondary labels in sample-submission order.

    Secondary labels are weak clip-level annotations, not 5-second ground truth.
    Soundscape annotations require their own interval-based target preparation.
    """
    if (
        not classes
        or any(
            not isinstance(label, str) or not label.strip() or label == ROW_ID for label in classes
        )
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Classes must be unique, nonempty species names, excluding row_id.")
    positions = {label: index for index, label in enumerate(classes)}
    targets = np.zeros((len(frame), len(classes)), dtype=np.float32)
    for row, (_, record) in enumerate(frame.iterrows()):
        labels = [record["primary_label"]]
        if include_secondary:
            secondary = record["secondary_labels"]
            if isinstance(secondary, str):
                try:
                    secondary = ast.literal_eval(secondary)
                except (ValueError, SyntaxError) as error:
                    raise ValueError("secondary_labels must be a list of label strings.") from error
            if not isinstance(secondary, list) or any(not isinstance(s, str) for s in secondary):
                raise ValueError("secondary_labels must be a list of label strings.")
            labels.extend(secondary)
        for label in labels:
            if label not in positions:
                raise ValueError(f"Unknown species label: {label!r}")
            targets[row, positions[label]] = 1
    return pd.DataFrame(targets, columns=classes, index=frame.index)


def align_targets(metadata: pd.DataFrame, targets: pd.DataFrame, *, key: str) -> pd.DataFrame:
    """Match keyed targets to metadata explicitly, including after CSV reload.

    Read saved targets with index_col=key. A RangeIndex cannot prove identity.
    Extra/missing keys, duplicates and null identifiers fail rather than guessing.
    """
    keys = metadata[key]
    if (
        targets.index.name != key
        or not targets.index.is_unique
        or targets.index.hasnans
        or keys.isna().any()
        or keys.duplicated().any()
        or set(keys) != set(targets.index)
    ):
        raise ValueError(f"Metadata and targets need the same unique, nonnull {key} keys.")
    result = targets.loc[keys].copy()
    result.index = metadata.index
    return result
