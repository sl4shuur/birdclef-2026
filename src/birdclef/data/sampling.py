"""Control the source mixture independently of dataset sizes and model weights."""

import math

import torch
from torch.utils.data import WeightedRandomSampler


class SourceWeightedRandomSampler(WeightedRandomSampler):
    """Sample ConcatDataset([focal, soundscapes]) with replacement.

    Probability is the expected soundscape fraction, not an exact batch quota.
    Updating it affects the next iterator/epoch; an active iterator has already
    drawn its indices. Validation must not use this sampler.
    """

    def __init__(
        self,
        focal_count: int,
        soundscape_count: int,
        *,
        soundscape_probability: float,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if focal_count <= 0 or soundscape_count <= 0:
            raise ValueError("Both source datasets must be nonempty.")
        self.focal_count = focal_count
        self.soundscape_count = soundscape_count
        generator = generator or torch.Generator().manual_seed(torch.initial_seed())
        self._seed = generator.initial_seed()
        size = focal_count + soundscape_count
        super().__init__(
            [1.0] * size,
            size if num_samples is None else num_samples,
            replacement=True,
            generator=generator,
        )
        self.set_soundscape_probability(soundscape_probability)

    def set_epoch(self, epoch: int) -> None:
        """Lightning forwards the epoch through its distributed sampler wrapper."""
        assert self.generator is not None
        self.generator.manual_seed(self._seed + epoch)

    def set_soundscape_probability(self, probability: float) -> None:
        """Call between epochs, including from a training callback if needed."""
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("soundscape_probability must be between zero and one.")
        self.weights[: self.focal_count] = (1 - probability) / self.focal_count
        self.weights[self.focal_count :] = probability / self.soundscape_count
        self.soundscape_probability = probability
