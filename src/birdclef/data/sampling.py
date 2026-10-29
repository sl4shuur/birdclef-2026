"""Control the source mixture independently of dataset sizes and model weights."""

import math
from collections.abc import Mapping, Sequence, Sized

import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler


class MixedSourceDataset(ConcatDataset):
    """Keep source membership alongside the actual concatenation order."""

    def __init__(self, datasets: Mapping[str, Dataset]) -> None:
        self.by_source = dict(datasets)
        super().__init__(list(self.by_source.values()))
        self.sources: list[str] = []
        for source, dataset in self.by_source.items():
            assert isinstance(dataset, Sized)
            self.sources.extend([source] * len(dataset))


class SourceWeightedRandomSampler(WeightedRandomSampler):
    """Sample by explicit source membership, independently of dataset ordering.

    Probability is the expected soundscape fraction, not an exact batch quota.
    Updating it affects the next iterator/epoch; an active iterator has already
    drawn its indices. Validation must not use this sampler.
    """

    def __init__(
        self,
        sources: Sequence[str],
        *,
        soundscape_probability: float,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        focal_count = sources.count("focal")
        soundscape_count = sources.count("soundscape")
        if focal_count <= 0 or soundscape_count <= 0:
            raise ValueError("Both source datasets must be nonempty.")
        if focal_count + soundscape_count != len(sources):
            raise ValueError("Expected focal and soundscape source names.")
        self.focal_count = focal_count
        self.soundscape_count = soundscape_count
        self._soundscape = torch.tensor([source == "soundscape" for source in sources])
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
        self.weights[~self._soundscape] = (1 - probability) / self.focal_count
        self.weights[self._soundscape] = probability / self.soundscape_count
        self.soundscape_probability = probability
