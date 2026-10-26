"""Deterministic sample transforms, independent of file loading and training."""

import librosa
import numpy as np
import torch

from birdclef.competition import SAMPLE_RATE


def log_mel(waveform: torch.Tensor) -> torch.Tensor:
    """32 kHz mono waveform -> (1, 128, frames), power dB re 1, 80 dB range.

    Matches the EDA candidate frontend. No fitted normalization or augmentation.
    This CPU transform is optional; models may instead consume raw waveforms.
    """
    if waveform.ndim != 1 or waveform.numel() < 1024:
        raise ValueError("Log-Mel requires a mono waveform with at least 1024 samples.")
    audio = waveform.detach().cpu().numpy()
    if not np.isfinite(audio).all():
        raise ValueError("Waveform must be finite.")
    power = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=320,
        n_mels=128,
        fmin=50,
        fmax=14000,
        power=2,
        center=False,
    )
    values = librosa.power_to_db(power, ref=1.0, top_db=80).astype(np.float32)
    return torch.from_numpy(values).unsqueeze(0)
