"""Encoder selection and settings, independent of the prediction head."""

from enum import StrEnum
from typing import Literal

from birdclef.config import ConfigModel, ProjectPath


class EncoderName(StrEnum):
    EFFICIENTNET = "efficientnet_b0"
    PERCH = "perch"


class EncoderConfig(ConfigModel):
    encoder: EncoderName = EncoderName.EFFICIENTNET
    encoder_path: ProjectPath | None = None
    encoder_device: Literal["cpu", "cuda"] = "cpu"
    pretrained: bool = True
