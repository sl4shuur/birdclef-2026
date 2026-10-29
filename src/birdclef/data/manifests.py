"""Prepare labeled units without reading audio or inventing missing annotations."""

from pathlib import Path

import numpy as np
import pandas as pd

from birdclef.competition import ROW_ID, SEGMENT_SECONDS, SOUNDSCAPE_SECONDS
from birdclef.data.labels import align_targets, encode_labels


def _audio_paths(directory: Path, filenames: pd.Series) -> list[str]:
    root = directory.resolve()
    paths = []
    for filename in filenames:
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Audio filenames must be nonempty strings.")
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Audio path escapes its directory: {filename}")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path))
    return paths


def focal_manifest(
    frame: pd.DataFrame, directory: Path, classes: list[str], *, include_secondary: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per weakly labeled clip; targets do not assert crop-level presence."""
    if frame.empty or frame.filename.duplicated().any():
        raise ValueError("Focal recordings must be nonempty with unique filenames.")
    metadata = frame.reset_index(drop=True).copy()
    metadata["path"] = _audio_paths(directory, metadata.filename)
    targets = encode_labels(metadata, classes, include_secondary=include_secondary)
    targets.index = pd.Index(metadata.filename, name="filename")
    return metadata, targets


def soundscape_manifest(
    frame: pd.DataFrame, directory: Path, classes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only explicitly annotated, aligned 5s windows. Conflicting labels fail.

    Repeated rows with the same species set collapse into one window. Missing or
    empty primary_label is not interpreted as an explicitly annotated negative.
    """
    annotations = frame[["filename", "start", "end", "primary_label"]].copy()
    if annotations.empty:
        raise ValueError("Soundscape annotations must be nonempty.")
    for column in ["start", "end"]:
        annotations[column] = pd.to_timedelta(
            annotations[column], errors="raise"
        ).dt.total_seconds()
    start, end = annotations.start, annotations.end
    if not (
        np.isfinite(start)
        & np.isfinite(end)
        & (start >= 0)
        & (end <= SOUNDSCAPE_SECONDS)
        & (end - start == SEGMENT_SECONDS)
        & (start % SEGMENT_SECONDS == 0)
    ).all():
        raise ValueError("Expected aligned 5-second intervals within a 60-second soundscape.")
    label_sets = []
    for value in annotations.primary_label:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Missing soundscape labels cannot be used as negative examples.")
        labels = value.split(";")
        if any(not label.strip() for label in labels):
            raise ValueError("Empty species code in soundscape annotations.")
        label_sets.append(tuple(sorted({label.strip() for label in labels})))
    annotations["labels"] = label_sets
    keys = ["filename", "start", "end"]
    if (annotations.groupby(keys, dropna=False).labels.nunique() > 1).any():
        raise ValueError("Conflicting label sets for the same soundscape window.")
    metadata = annotations.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    metadata["path"] = _audio_paths(directory, metadata.filename)
    metadata[ROW_ID] = [
        f"{Path(name).stem}_{int(end)}"
        for name, end in zip(metadata.filename, metadata.end, strict=True)
    ]
    if metadata[ROW_ID].duplicated().any():
        raise ValueError("Soundscape row IDs must be unique.")
    # Keep one implementation of species validation and multi-hot class ordering.
    label_frame = pd.DataFrame(
        {
            "primary_label": [labels[0] for labels in metadata.labels.tolist()],
            "secondary_labels": [list(labels[1:]) for labels in metadata.labels.tolist()],
        }
    )
    targets = encode_labels(label_frame, classes)
    targets.index = pd.Index(metadata[ROW_ID], name=ROW_ID)
    return metadata.drop(columns=["primary_label", "labels"]), targets


def filter_focal_duration(
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    headers: pd.DataFrame,
    *,
    min_duration: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exclude clips below the threshold, keeping an explicit review table.

    Headers must cover every focal filename once. Source audio stays untouched.
    """
    if not np.isfinite(min_duration) or min_duration < 0:
        raise ValueError("min_duration must be finite and nonnegative.")
    align_targets(metadata, targets, key="filename")
    durations = headers.set_index("filename").duration
    if (
        not durations.index.is_unique
        or durations.index.hasnans
        or set(durations.index) != set(metadata.filename)
    ):
        raise ValueError("Headers must cover the same unique focal filenames.")
    durations = durations.reindex(metadata.filename).to_numpy(dtype=float)
    if not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError("Audio durations must be finite and positive.")
    keep = durations >= min_duration
    excluded = metadata.loc[~keep].copy()
    excluded["duration"] = durations[~keep]
    retained = metadata.loc[keep].copy()
    if retained.empty:
        raise ValueError("Duration filtering removed every focal recording.")
    return retained, targets.loc[retained.filename].copy(), excluded
