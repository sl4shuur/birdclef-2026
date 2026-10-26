import pytest

from birdclef.config import LoaderConfig, LRSchedulerConfig, TrainConfig
from birdclef.models import AudioExperimentConfig, AudioModelConfig


def test_run_name_composes_model_split_and_training_settings():
    config = AudioExperimentConfig(
        model=AudioModelConfig(target_columns=["bird"]),
        loader=LoaderConfig(batch_size=4, soundscape_probability=0.1),
        train=TrainConfig(seed=42),
        lr_scheduler=LRSchedulerConfig(name="none"),
        fold=2,
        smoke=True,
    )
    assert config.run_name() == (
        "audio-cnn__fold2__effb0-imagenet-drop0.2__bs4-lr0.0003-s42__wd0.01-sc0.1__none__smoke"
    )
    original = config.run_name()
    config.train.max_epochs += 5
    assert config.run_name() == original
    config.model.pretrained = False
    assert "effb0-scratch" in config.run_name()
    config.model.learning_rate = 0.001
    assert "lr0.001" in config.run_name()
    config.train.accumulate_grad_batches = 2
    assert "acc2" in config.run_name()
    config.loader.soundscape_probability = 0.3
    assert "sc0.3" in config.run_name()
    config.smoke = False
    assert not config.run_name().endswith("__smoke")


@pytest.mark.parametrize("name", ["", "../run", r"..\run", "a:b", ".", "run."])
def test_rejects_invalid_run_name_prefix(name):
    config = AudioExperimentConfig(model=AudioModelConfig(target_columns=["bird"]))
    with pytest.raises(ValueError, match="Run name"):
        config.run_name(name)
