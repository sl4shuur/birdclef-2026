"""Shared data domain models"""

from hashlib import sha256
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Higher ranks first
    score: float

    @field_validator("score")
    @classmethod
    def _not_nan(cls, score: float) -> float:
        if np.isnan(score):
            raise ValueError("score must not be NaN.")
        return score


class SplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    dataset_id: str
    seed: int
    train_keys: list[str]
    val_keys: list[str]
    train_rows: int
    val_rows: int
    row_keys_checksum: str

    @model_validator(mode="after")
    def _partitions(self) -> Self:
        if not self.dataset_id or not self.train_keys or not self.val_keys:
            raise ValueError("Dataset identity and both split partitions must be nonempty.")
        if set(self.train_keys) & set(self.val_keys):
            raise ValueError("Train and validation groups overlap.")
        if len(set(self.train_keys)) != len(self.train_keys) or len(set(self.val_keys)) != len(
            self.val_keys
        ):
            raise ValueError("Split group keys must be unique.")
        if self.train_rows < len(self.train_keys) or self.val_rows < len(self.val_keys):
            raise ValueError("Each group needs at least one row.")
        return self

    @property
    def checksum(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()


class DataSplit(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    train: pd.DataFrame
    val: pd.DataFrame
    manifest: SplitManifest


type Phase = Literal["training", "evaluating", "complete", "interrupted", "failed"]


class RunState(BaseModel):
    """Saved as `state.json`: which run a folder belongs to, and how far it got."""

    version: Literal[1] = 1
    run_id: str
    fingerprint: str
    experiment_name: str
    phase: Phase = "training"
