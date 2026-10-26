"""Prepare labeled units without reading audio or inventing missing annotations."""

from pathlib import Path

import numpy as np
import pandas as pd

from birdclef.competition import ROW_ID, SEGMENT_SECONDS, SOUNDSCAPE_SECONDS
from birdclef.data.labels import encode_labels


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
            "primary_label": metadata.labels.map(lambda labels: labels[0]),
            "secondary_labels": metadata.labels.map(lambda labels: list(labels[1:])),
        }
    )
    targets = encode_labels(label_frame, classes)
    return metadata.drop(columns=["primary_label", "labels"]), targets
