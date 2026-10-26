"""Read the saved preparation artifacts without recomputing labels or folds."""

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
from torch.utils.data import ConcatDataset

from birdclef.data.audio import FocalAudioDataset, SoundscapeDataset
from birdclef.data.labels import align_targets
from birdclef.processing.audio import log_mel


def load_audio_split(
    prepared_dir: Path, competition_dir: Path, *, fold: int, classes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify artifact identity, rebuild paths and select an existing validation fold."""
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != 2:
        raise ValueError("Regenerate preparation artifacts with keyed targets (version 2).")
    for name in [
        "classes.json",
        "focal.csv",
        "focal_targets.csv",
        "soundscapes.csv",
        "soundscape_targets.csv",
    ]:
        actual = sha256((prepared_dir / name).read_bytes()).hexdigest()
        if actual != manifest["artifacts_sha256"][name]:
            raise ValueError(f"Preparation artifact checksum mismatch: {name}")
    if json.loads((prepared_dir / "classes.json").read_text(encoding="utf-8")) != classes:
        raise ValueError("Model classes must match the saved preparation class order.")
    frames = []
    for source, name, target_name, key, audio_dir in [
        ("focal", "focal.csv", "focal_targets.csv", "filename", "train_audio"),
        ("soundscape", "soundscapes.csv", "soundscape_targets.csv", "row_id", "train_soundscapes"),
    ]:
        metadata = pd.read_csv(prepared_dir / name)
        targets = pd.read_csv(prepared_dir / target_name, index_col=key)
        if targets.columns.tolist() != classes:
            raise ValueError(f"Target class order mismatch: {target_name}")
        frame = pd.concat([metadata, align_targets(metadata, targets, key=key)], axis=1)
        frame["path"] = frame.filename.map(
            lambda value, folder=audio_dir: str(competition_dir / folder / value)
        )
        frame["source"] = source
        frames.append(frame)
    focal, soundscapes = frames
    if not soundscapes.groupby("filename").fold.nunique().eq(1).all():
        raise ValueError("A recording occurs in multiple saved folds.")
    valid = soundscapes.loc[soundscapes.fold.eq(fold)].copy()
    training_sc = soundscapes.loc[soundscapes.fold.ne(fold)].copy()
    if valid.empty or training_sc.empty:
        raise ValueError("Select a saved fold with nonempty training and validation windows.")
    return pd.concat([focal, training_sc], ignore_index=True), valid.reset_index(drop=True)


def audio_training_datasets(
    train: pd.DataFrame, valid: pd.DataFrame, *, classes: list[str]
) -> tuple[ConcatDataset, SoundscapeDataset]:
    """Mixed-source training and soundscape-only validation, with no recording leakage."""
    if train.empty or valid.empty:
        raise ValueError("Training and validation must be nonempty.")
    if (
        not train.source.isin(["focal", "soundscape"]).all()
        or not valid.source.eq("soundscape").all()
    ):
        raise ValueError("Expected focal/soundscape training and soundscape-only validation.")
    focal = train.loc[train.source.eq("focal")]
    soundscapes = train.loc[train.source.eq("soundscape")]
    if focal.empty or soundscapes.empty:
        raise ValueError("Both training sources must be nonempty.")
    if set(soundscapes.filename) & set(valid.filename):
        raise ValueError("Training and validation soundscape recordings overlap.")
    focal_dataset = FocalAudioDataset(
        focal, focal.set_index("filename")[classes], training=True, transform=log_mel
    )
    sc_dataset = SoundscapeDataset(
        soundscapes, soundscapes.set_index("row_id")[classes], transform=log_mel
    )
    validation = SoundscapeDataset(valid, valid.set_index("row_id")[classes], transform=log_mel)
    return ConcatDataset([focal_dataset, sc_dataset]), validation
