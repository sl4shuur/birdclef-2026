import pytest

torch = pytest.importorskip("torch")

from birdclef.config import LoaderConfig  # noqa: E402
from birdclef.data.sampling import MixedSourceDataset, SourceWeightedRandomSampler  # noqa: E402


def test_source_mass_and_updates_apply_on_next_iterator():
    sampler = SourceWeightedRandomSampler(
        ["focal"] * 100 + ["soundscape"] * 2, soundscape_probability=0.3, num_samples=30
    )
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
                ["focal"] * 100 + ["soundscape"] * 2,
                soundscape_probability=0.5,
                generator=torch.Generator().manual_seed(42),
            )
        )

    assert draw() == draw()


def test_ddp_shards_the_same_weighted_draw_and_resumes_its_epoch():
    from lightning.fabric.utilities.distributed import DistributedSamplerWrapper

    def sampler():
        return SourceWeightedRandomSampler(
            ["focal"] * 100 + ["soundscape"] * 2,
            soundscape_probability=0.3,
            num_samples=1002,
            generator=torch.Generator().manual_seed(42),
        )

    expected = sampler()
    expected.set_epoch(3)
    full = list(expected)
    shards = []
    for rank in range(2):
        shard = DistributedSamplerWrapper(sampler(), num_replicas=2, rank=rank, seed=42)
        shard.set_epoch(3)
        values = list(shard)
        assert len(values) == 501
        shards.extend(values)
    assert sorted(shards) == sorted(full)
    expected.set_epoch(3)
    assert list(expected) == full
    expected.set_epoch(4)
    assert list(expected) != full


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_probability(probability):
    with pytest.raises(ValueError):
        SourceWeightedRandomSampler(
            ["focal"] * 3 + ["soundscape"] * 2, soundscape_probability=probability
        )
    with pytest.raises(ValueError):
        LoaderConfig(soundscape_probability=probability)


def test_datamodule_uses_sampler_and_leaves_validation_sequential():
    pytest.importorskip("lightning")
    from birdclef.data.datamodule import TrainingDataModule

    dataset = torch.utils.data.TensorDataset(torch.arange(5))
    sampler = SourceWeightedRandomSampler(
        ["focal"] * 3 + ["soundscape"] * 2, soundscape_probability=1, num_samples=8
    )
    module = TrainingDataModule(
        dataset, dataset, config=LoaderConfig(batch_size=8), sampler=sampler
    )
    loader = module.train_dataloader()
    assert loader.sampler is sampler
    assert all(value >= 3 for value in next(iter(loader))[0].tolist())
    sampler.set_soundscape_probability(0)
    assert all(value < 3 for value in next(iter(loader))[0].tolist())
    assert next(iter(module.val_dataloader()))[0].tolist() == list(range(5))


@pytest.mark.parametrize("reverse", [False, True])
def test_classifier_samples_the_requested_source_regardless_of_dataset_order(reverse):
    pytest.importorskip("lightning")
    from birdclef.models import AudioClassifier, AudioExperimentConfig

    sources = {
        "focal": torch.utils.data.TensorDataset(torch.zeros(3)),
        "soundscape": torch.utils.data.TensorDataset(torch.ones(2)),
    }
    if reverse:
        sources = dict(reversed(list(sources.items())))
    dataset = MixedSourceDataset(sources)
    config = AudioExperimentConfig(loader=LoaderConfig(batch_size=5, soundscape_probability=1))
    model = AudioClassifier(config, classes=["bird"])
    data = model._training_loader(dataset, dataset)
    assert next(iter(data.train_dataloader()))[0].tolist() == [1.0] * 5
    assert model.sampler is not None
    model.sampler.set_soundscape_probability(0)
    assert next(iter(data.train_dataloader()))[0].tolist() == [0.0] * 5


def test_sampler_rejects_unknown_sources():
    with pytest.raises(ValueError, match="source names"):
        SourceWeightedRandomSampler(["focal", "soundscape", "unknown"], soundscape_probability=0.5)
