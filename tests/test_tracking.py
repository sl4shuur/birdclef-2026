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


def test_fit_cli_passes_existing_wandb_wrapper_and_environment_secret(tmp_path, monkeypatch):
    from birdclef.models.baseline.fit import fit
    from birdclef.models.baseline.model import Baseline

    tracking_module = importlib.import_module("birdclef.loggers.wandb")
    tracker = Mock()
    received = {}

    @contextmanager
    def wandb_run(config, *, name, save_dir):
        received.update(config=config, name=name, save_dir=save_dir)
        yield tracker

    monkeypatch.setattr(tracking_module, "wandb_run", wandb_run)
    monkeypatch.setattr("birdclef.loggers.apply_logging_config", Mock())
    monkeypatch.setattr("birdclef.loggers.CustomLogger", Mock())
    train = Mock()
    monkeypatch.setattr(Baseline, "fit", train)
    monkeypatch.setenv("KS_TRACKING__API_KEY", "secret-from-environment")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"output_dir": str(tmp_path / "outputs")},
                "tracking": {"offline": True},
            }
        )
    )
    data = tmp_path / "data.csv"
    data.write_text("feature,target\n1,2\n")
    result = CliRunner().invoke(
        fit,
        [
            "--config",
            str(config_path),
            "--name",
            "example",
            "--dataset-id",
            "data-v1",
            "--train",
            str(data),
            "--valid",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output
    assert received["config"].api_key.get_secret_value() == "secret-from-environment"
    assert received["save_dir"] == tmp_path / "outputs" / "example"
    assert train.call_args.args[2].tracker is tracker
    assert "secret-from-environment" not in result.output
