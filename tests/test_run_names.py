import pytest

from birdclef.models import AudioExperimentConfig


def test_run_name_keeps_the_experiment_path_and_separates_smoke_checks():
    config = AudioExperimentConfig()
    name = "audio-cnn/linear-probing"
    assert config.run_name(name) == name
    config.train.max_epochs += 5
    config.model.head_lr = 0.001
    assert config.run_name(name) == name
    config.smoke = True
    assert config.run_name(name) == f"{name}/smoke"


@pytest.mark.parametrize(
    "name", ["", "../run", r"..\run", "a:b", ".", "run.", "/run", "a//b", "a/../b", "run/"]
)
def test_rejects_invalid_experiment_paths(name):
    with pytest.raises(ValueError, match="Run name"):
        AudioExperimentConfig().run_name(name)
