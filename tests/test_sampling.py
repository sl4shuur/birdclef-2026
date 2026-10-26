import pytest

torch = pytest.importorskip("torch")

from birdclef.config import LoaderConfig  # noqa: E402
from birdclef.data.sampling import SourceWeightedRandomSampler  # noqa: E402


def test_source_mass_and_updates_apply_on_next_iterator():
    sampler = SourceWeightedRandomSampler(100, 2, soundscape_probability=0.3, num_samples=30)
    assert sampler.weights[:100].sum().item() == pytest.approx(0.7)
    assert sampler.weights[100:].sum().item() == pytest.approx(0.3)
    sampler.set_soundscape_probability(1)
    assert all(index >= 100 for index in sampler)
    sampler.set_soundscape_probability(0)
    assert all(index < 100 for index in sampler)


def test_sampler_reproducibility():
    def draw():
        return list(
            SourceWeightedRandomSampler(
                100, 2, soundscape_probability=0.5, generator=torch.Generator().manual_seed(42)
            )
        )

    assert draw() == draw()


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_probability(probability):
    with pytest.raises(ValueError):
        SourceWeightedRandomSampler(3, 2, soundscape_probability=probability)
    with pytest.raises(ValueError):
        LoaderConfig(soundscape_probability=probability)


def test_datamodule_uses_sampler_and_leaves_validation_sequential():
    pytest.importorskip("lightning")
    from birdclef.data.datamodule import TrainingDataModule

    dataset = torch.utils.data.TensorDataset(torch.arange(5))
    sampler = SourceWeightedRandomSampler(3, 2, soundscape_probability=1, num_samples=8)
    module = TrainingDataModule(
        dataset, dataset, config=LoaderConfig(batch_size=8), sampler=sampler
    )
    loader = module.train_dataloader()
    assert loader.sampler is sampler
    assert all(value >= 3 for value in next(iter(loader))[0].tolist())
    sampler.set_soundscape_probability(0)
    assert all(value < 3 for value in next(iter(loader))[0].tolist())
    assert next(iter(module.val_dataloader()))[0].tolist() == list(range(5))
