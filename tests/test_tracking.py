import importlib
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from pydantic import SecretStr

from birdclef.config import TrackingConfig


@pytest.mark.parametrize("fail", [False, True])
def test_wandb_run_reuses_saved_identity_and_finishes(tmp_path, monkeypatch, fail):
    module = importlib.import_module("birdclef.loggers.wandb")
    fake_wandb = SimpleNamespace(run=None, login=Mock())
    monkeypatch.setattr(module, "wandb", fake_wandb)
    opened = []

    def open_run(config, run_id, name, save_dir):
        run = Mock()
        run.finish.side_effect = lambda **kwargs: setattr(fake_wandb, "run", None)
        fake_wandb.run = run
        tracker = Mock()
        opened.append((run_id, run, tracker))
        return tracker

    monkeypatch.setattr(module, "_open", open_run)
    config = TrackingConfig(offline=True, api_key=SecretStr("unused-offline-token"))
    with module.wandb_run(config, name="example", save_dir=tmp_path) as tracker:
        assert tracker is opened[-1][2]
    identity = (tmp_path / "wandb-run-id.txt").read_text()
    assert opened[0][0] == identity
    opened[0][1].finish.assert_called_once_with(exit_code=0)

    def second_launch():
        with module.wandb_run(config, name="example", save_dir=tmp_path):
            if fail:
                raise RuntimeError("training failed")

    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            second_launch()
    else:
        second_launch()
    assert opened[1][0] == identity
    opened[1][1].finish.assert_called_once_with(exit_code=int(fail))
    fake_wandb.login.assert_not_called()


def test_wandb_prefers_process_secret_and_closes_initialization_failure(tmp_path, monkeypatch):
    module = importlib.import_module("birdclef.loggers.wandb")
    run = Mock()
    fake_wandb = SimpleNamespace(run=None, login=Mock())
    monkeypatch.setattr(module, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "process-secret")

    def fail_open(*args, **kwargs):
        fake_wandb.run = run
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(module, "_open", fail_open)
    config = TrackingConfig(offline=False, api_key=SecretStr("config-secret"))
    with pytest.raises(RuntimeError, match="initialization failed"):
        with module.wandb_run(config, name="example", save_dir=tmp_path):
            pass
    fake_wandb.login.assert_called_once_with(key="process-secret")
    run.finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize("rank", [0, 1])
def test_audio_fit_cli_uses_wandb_and_saves_secret_free_artifacts(tmp_path, monkeypatch, rank):
    import pandas as pd
    import wandb
    from lightning.pytorch.utilities import rank_zero_only

    from birdclef.models import AudioClassifier, AudioExperimentConfig
    from birdclef.models.classifier.fit import fit

    tracker = Mock()
    monkeypatch.setattr(rank_zero_only, "rank", rank)
    tracker.experiment.id = "test-run"
    artifact = Mock()
    monkeypatch.setattr(wandb, "Artifact", Mock(return_value=artifact))
    monkeypatch.setattr("birdclef.loggers.apply_logging_config", Mock())
    monkeypatch.setattr("birdclef.loggers.CustomLogger", Mock())
    monkeypatch.setenv("BCLEF_TRACKING__API_KEY", "environment-secret")
    received = {}

    @contextmanager
    def fake_run(config, *, name, save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)
        received.update(config=config, name=name, directory=save_dir)
        yield tracker

    monkeypatch.setattr("birdclef.loggers.wandb_run", fake_run)
    training = pd.DataFrame({"source": ["focal"] * 40 + ["soundscape"] * 40})
    validation = pd.DataFrame({"row_id": ["a_5", "b_5"]})
    load_split = Mock(return_value=(training, validation))
    monkeypatch.setattr("birdclef.data.load_audio_split", load_split)

    def fake_fit(self, train, valid, context):
        received.update(train=train, valid=valid, context=context)
        directory = context.output_dir
        (directory / "checkpoints").mkdir()
        (directory / "checkpoints/best.ckpt").write_bytes(b"test checkpoint")
        (directory / "model-config.json").write_text(json.dumps(self.config.safe_dump()))
        (directory / "classes.json").write_text(json.dumps(self.classes))
        (directory / "fit-inputs.json").write_text("{}")
        (directory / "fit.json").write_text(
            json.dumps({"best_checkpoint": "checkpoints/best.ckpt"})
        )
        (directory / "validation-metrics.json").write_text(json.dumps({"auc": 1.0}))
        (directory / "validation-predictions.csv").write_text("row_id,bird\na_5,0.9\n")
        (directory / "validation-class-metrics.csv").write_text(
            "species,positive,negative,auc\nbird,1,1,1\n"
        )

    monkeypatch.setattr(AudioClassifier, "fit", fake_fit)
    payload = {
        "data": {"output_dir": str(tmp_path / "outputs")},
        "model": {},
        "tracking": {"offline": True},
        "smoke": True,
        "fold": 2,
    }
    config_path = tmp_path / "launch.json"
    config_path.write_text(json.dumps(payload))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "classes.json").write_text(json.dumps(["bird"]))
    result = CliRunner().invoke(
        fit,
        [
            "--config",
            str(config_path),
            "--name",
            "audio-cnn/probe",
            "--prepared-dir",
            str(prepared),
        ],
    )
    assert result.exit_code == 0, result.output
    expected = AudioExperimentConfig.model_validate(payload).run_name("audio-cnn/probe")
    assert received["name"] == expected
    assert received["directory"] == tmp_path / "outputs" / "audio-cnn" / "probe" / "smoke"
    assert received["config"].api_key.get_secret_value() == "environment-secret"
    assert received["train"].source.value_counts().to_dict() == {"focal": 32, "soundscape": 32}
    assert received["valid"] is validation
    assert received["context"].tracker is tracker
    assert received["context"].protocol == "file-gkf5/fold-2/smoke"
    assert "environment-secret" not in result.output
    assert "environment-secret" not in (received["directory"] / "model-config.json").read_text()
    if rank != 0:
        tracker.log_table.assert_not_called()
        tracker.experiment.log_artifact.assert_not_called()
        return
    tracker.log_table.assert_called_once()
    tracker.experiment.log_artifact.assert_called_once_with(artifact, aliases=["best"])
    names = {call.kwargs["name"] for call in artifact.add_file.call_args_list}
    assert "checkpoints/best.ckpt" in names
    assert "model-config.json" in names
    assert "classes.json" in names
