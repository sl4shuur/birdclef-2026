"""Complete recording context, masked targets and padded focal sequences."""

from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from birdclef.competition import ROW_ID, SEGMENT_SECONDS, SOUNDSCAPE_SECONDS
from birdclef.data.features import feature_tensor

SEQUENCE_LENGTH = SOUNDSCAPE_SECONDS // SEGMENT_SECONDS
WINDOW_KEY = ["filename", "start"]


def recording_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand only the supplied recordings, without adding labels to missing windows."""
    if frame.empty or frame.duplicated(WINDOW_KEY).any():
        raise ValueError("Expected nonempty, unique soundscape windows.")
    if "source" in frame and not frame.source.eq("soundscape").all():
        raise ValueError("Recording context requires soundscape windows.")
    if not (
        frame.start.ge(0)
        & frame.end.le(SOUNDSCAPE_SECONDS)
        & frame.start.mod(SEGMENT_SECONDS).eq(0)
        & (frame.end - frame.start).eq(SEGMENT_SECONDS)
    ).all():
        raise ValueError("Expected aligned windows within a 60-second recording.")
    if not frame.groupby("filename").path.nunique().eq(1).all():
        raise ValueError("Each recording must have one audio path.")
    recordings = frame[["filename", "path"]].drop_duplicates().sort_values("filename")
    offsets = pd.DataFrame({"start": range(0, SOUNDSCAPE_SECONDS, SEGMENT_SECONDS)})
    windows = recordings.merge(offsets, how="cross")
    windows["end"] = windows.start + SEGMENT_SECONDS
    windows[ROW_ID] = [
        f"{Path(name).stem}_{end}" for name, end in zip(windows.filename, windows.end, strict=True)
    ]
    windows["source"] = "soundscape"
    return windows


def window_targets(
    windows: pd.DataFrame, labeled: pd.DataFrame, classes: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Missing annotations become mask=False, never supervised negative targets."""
    values = feature_tensor(labeled, classes)
    if not ((values == 0) | (values == 1)).all() or labeled.duplicated(WINDOW_KEY).any():
        raise ValueError("Expected unique windows with binary targets.")
    targets = labeled.set_index(WINDOW_KEY)[classes].reindex(
        pd.MultiIndex.from_frame(windows[WINDOW_KEY])
    )
    mask = torch.tensor(targets.notna().all(axis=1).to_numpy())
    return feature_tensor(targets.fillna(0), classes), mask


class SequenceDataset(TensorDataset):
    """Pad on access, keeping focal embeddings stored only once in RAM."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        inputs, targets, mask, length = super().__getitem__(index)
        padding = SEQUENCE_LENGTH - len(inputs)
        return (
            F.pad(inputs, (0, 0, 0, padding)),
            F.pad(targets, (0, 0, 0, padding)),
            F.pad(mask, (0, padding)),
            length,
        )


def sequence_dataset(
    inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, *, length: int
) -> TensorDataset:
    """Group ordered windows; pad focal length-1 examples for mixed-source batches."""
    inputs = inputs.reshape(-1, length, inputs.shape[-1])
    targets = targets.reshape(-1, length, targets.shape[-1])
    mask = mask.reshape(-1, length)
    if not mask.any(dim=1).all():
        raise ValueError("Every training sequence needs at least one labeled position.")
    return SequenceDataset(
        inputs,
        targets,
        mask,
        torch.full((len(inputs),), length, dtype=torch.long),
    )


def requested_indices(frame: pd.DataFrame, windows: pd.DataFrame) -> torch.Tensor:
    """Map flattened sequence scores back to the caller's window order."""
    indices = pd.MultiIndex.from_frame(windows[WINDOW_KEY]).get_indexer(
        pd.MultiIndex.from_frame(frame[WINDOW_KEY])
    )
    if (indices < 0).any():
        raise ValueError("Requested windows are missing from recording context.")
    return torch.from_numpy(indices)
