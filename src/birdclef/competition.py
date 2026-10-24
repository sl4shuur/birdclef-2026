"""BirdCLEF+ 2026 conventions shared by data preparation and inference."""

from pathlib import Path

import pandas as pd

COMPETITION = "birdclef-2026"
SAMPLE_RATE = 32_000
SEGMENT_SECONDS = 5
SOUNDSCAPE_SECONDS = 60
ROW_ID = "row_id"


def class_columns(sample: pd.DataFrame) -> list[str]:
    """Use the official sample's column order, never sorted taxonomy labels."""
    columns = sample.columns.tolist()
    if not columns or columns[0] != ROW_ID or len(columns) < 2:
        raise ValueError("Expected row_id followed by species columns.")
    if not sample.columns.is_unique or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("Submission columns must be unique, nonempty strings.")
    return columns[1:]


def soundscape_windows(directory: Path) -> pd.DataFrame:
    """Manifest for official 60-second OGG files; no audio decoder is needed.

    Hidden test rows must come from the files present at inference time, not the
    public sample's placeholder IDs. Empty test directories fail explicitly.
    """
    if not directory.is_dir():
        raise ValueError(f"Soundscape directory does not exist: {directory}")
    files = sorted(directory.glob("*.ogg"))
    if not files:
        raise ValueError(f"No OGG soundscapes found in {directory}.")
    return pd.DataFrame(
        [
            {
                ROW_ID: f"{path.stem}_{end}",
                "filename": path.name,
                "path": str(path),
                "start": end - SEGMENT_SECONDS,
                "end": end,
            }
            for path in files
            for end in range(SEGMENT_SECONDS, SOUNDSCAPE_SECONDS + 1, SEGMENT_SECONDS)
        ]
    )
