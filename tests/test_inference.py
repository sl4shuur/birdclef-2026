import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from birdclef.submission import predict_runs, run_inference
from birdclef.submission.export import main as export_models


def test_ensemble_aligns_rows_and_classes_and_overrides_training_hardware(monkeypatch):
    pytest.importorskip("lightning")
    from birdclef.models import AudioClassifier

    frames = {
        "a": pd.DataFrame({"row_id": ["x_5", "x_10"], "a": [0.2, 0.4], "b": [0.8, 0.6]}),
        "b": pd.DataFrame({"row_id": ["x_10", "x_5"], "b": [0.2, 0.4], "a": [0.8, 0.6]}),
    }

    class Model:
        def __init__(self, path):
            self.path = path
            self.config = SimpleNamespace(runtime=None, loader=None)

        def predict(self, windows):
            assert self.config.runtime.model_dump() == {
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
            }
            assert self.config.loader.predict_batch_size == 8
            assert self.config.loader.num_workers == 0
            assert not self.config.loader.pin_memory
            return frames[self.path.name]

    monkeypatch.setattr(AudioClassifier, "from_run", Model)
    windows = pd.DataFrame({"row_id": ["x_5", "x_10"]})
    template = windows.assign(a=0.0, b=0.0)
    result = predict_runs([Path("a"), Path("b")], windows, template, weights=[1, 3], batch_size=8)
    assert result.columns.tolist() == ["row_id", "a", "b"]
    assert result.row_id.tolist() == windows.row_id.tolist()
    np.testing.assert_allclose(result[["a", "b"]], [[0.5, 0.5], [0.7, 0.3]])


@pytest.mark.parametrize("weights", [[1], [0, 1], [-1, 1], [float("nan"), 1], [float("inf"), 1]])
def test_ensemble_rejects_invalid_weights_before_loading_models(weights):
    pytest.importorskip("lightning")
    with pytest.raises(ValueError, match="weight"):
        predict_runs(
            [Path("missing-a"), Path("missing-b")], pd.DataFrame(), pd.DataFrame(), weights=weights
        )


@pytest.mark.parametrize("hidden", [False, True])
def test_public_preview_checks_real_audio_but_hidden_submission_uses_actual_ids(
    tmp_path, monkeypatch, hidden
):
    import birdclef.submission.predict as inference

    competition = tmp_path / "competition"
    train = competition / "train_soundscapes"
    test = competition / "test_soundscapes"
    train.mkdir(parents=True)
    test.mkdir()
    (train / "train.ogg").touch()
    if hidden:
        (test / "test_b.ogg").touch()
        (test / "test_a.ogg").touch()
    sample = pd.DataFrame({"row_id": ["placeholder_5"], "a": [0.0], "b": [0.0]})
    sample.to_csv(competition / "sample_submission.csv", index=False)
    seen = []

    def predict(runs, windows, template, **kwargs):
        seen.extend(windows.path)
        return pd.DataFrame({"row_id": windows.row_id, "a": 0.2, "b": 0.8})

    monkeypatch.setattr(inference, "predict_runs", predict)
    output = tmp_path / "result" / "submission.csv"
    assert run_inference([Path("model")], competition, output) == output
    result = pd.read_csv(output)
    if hidden:
        assert len(result) == 24
        assert result.row_id.tolist() == [
            f"test_{name}_{end}" for name in ("a", "b") for end in range(5, 61, 5)
        ]
        assert all(Path(path).parent == test for path in seen)
        assert not output.with_name("smoke-predictions.csv").exists()
    else:
        pd.testing.assert_frame_equal(result, sample)
        smoke = pd.read_csv(output.with_name("smoke-predictions.csv"))
        assert smoke.row_id.tolist() == [f"train_{end}" for end in range(5, 61, 5)]
        assert all(Path(path).parent == train for path in seen)


def test_explicit_empty_audio_folder_fails_without_placeholder(tmp_path):
    pd.DataFrame({"row_id": ["placeholder_5"], "a": [0.0]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    audio = tmp_path / "empty"
    audio.mkdir()
    output = tmp_path / "submission.csv"
    with pytest.raises(ValueError, match="No OGG"):
        run_inference([Path("model")], tmp_path, output, soundscapes=audio)
    assert not output.exists()


def test_export_keeps_only_best_checkpoint_and_portable_metadata(tmp_path):
    pytest.importorskip("lightning")
    from birdclef.models import AudioExperimentConfig

    run = tmp_path / "model"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.ckpt").write_bytes(b"best checkpoint")
    (run / "checkpoints" / "last.ckpt").write_bytes(b"not needed")
    (run / "classes.json").write_text(json.dumps(["a", "b"]))
    (run / "model-config.json").write_text(
        json.dumps(AudioExperimentConfig().safe_dump()), encoding="utf-8"
    )
    (run / "fit.json").write_text(json.dumps({"best_checkpoint": "checkpoints/best.ckpt"}))
    output = tmp_path / "models.zip"
    result = CliRunner().invoke(
        export_models, ["--run", str(run), "--all", "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    with ZipFile(output) as archive:
        assert {name for name in archive.namelist() if not name.endswith("/")} == {
            "model/model-config.json",
            "model/classes.json",
            "model/fit.json",
            "model/checkpoints/best.ckpt",
        }
        assert archive.read("model/checkpoints/best.ckpt") == b"best checkpoint"
        assert "api_key" not in archive.read("model/model-config.json").decode()


@pytest.fixture
def export_run(tmp_path):
    from birdclef.models import AudioExperimentConfig

    def create(name, *, completed=False):
        run = tmp_path / name
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints/last.ckpt").write_bytes(b"last checkpoint")
        (run / "classes.json").write_text(json.dumps(["a", "b"]))
        (run / "model-config.json").write_text(
            json.dumps(AudioExperimentConfig().safe_dump()), encoding="utf-8"
        )
        if completed:
            (run / "checkpoints/best.ckpt").write_bytes(b"best checkpoint")
            (run / "fit.json").write_text(json.dumps({"best_checkpoint": "checkpoints/best.ckpt"}))
        return run

    return create


def test_export_discovers_nested_runs_deduplicates_roots_and_normalizes_interrupted_metadata(
    tmp_path, export_run
):
    from birdclef.models import read_fit_state

    interrupted = export_run("models/paused")
    completed = export_run("models/done", completed=True)
    incomplete = tmp_path / "models/incomplete"
    incomplete.mkdir()
    (incomplete / "model-config.json").write_text("{}")
    output = tmp_path / "models.zip"
    result = CliRunner().invoke(
        export_models,
        [
            "--run",
            str(tmp_path / "models"),
            "--run",
            str(interrupted),
            "--all",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Skipping" in result.output
    with ZipFile(output) as archive:
        assert archive.read("paused/checkpoints/last.ckpt") == b"last checkpoint"
        assert archive.read("done/checkpoints/best.ckpt") == b"best checkpoint"
        assert (
            json.loads(archive.read("paused/fit.json"))["best_checkpoint"]
            == "checkpoints/last.ckpt"
        )
        assert len([name for name in archive.namelist() if name.endswith("model-config.json")]) == 2
        archive.extractall(tmp_path / "unpacked")
    restored = tmp_path / "unpacked/paused"
    assert read_fit_state(restored).best_path(restored).read_bytes() == b"last checkpoint"
    assert not (interrupted / "fit.json").exists()
    assert read_fit_state(completed).best_checkpoint == "checkpoints/best.ckpt"


def test_export_real_checkbox_selects_only_the_checked_model(tmp_path, export_run, monkeypatch):
    import questionary
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    export_run("a")
    export_run("b")
    checkbox = questionary.checkbox
    labels = []
    with create_pipe_input() as pipe:

        def prompt(message, **kwargs):
            labels.extend(choice.title for choice in kwargs["choices"])
            return checkbox(message, **kwargs, input=pipe, output=DummyOutput())

        monkeypatch.setattr(questionary, "checkbox", prompt)
        pipe.send_text("\x1b[B \r")  # Down, Space, Enter: select only the second run.
        output = tmp_path / "selected.zip"
        result = CliRunner().invoke(
            export_models, ["--run", str(tmp_path), "--output", str(output)]
        )
    assert result.exit_code == 0, result.output
    assert all("last.ckpt" in label for label in labels)
    with ZipFile(output) as archive:
        assert {name.split("/")[0] for name in archive.namelist()} == {"b"}


@pytest.mark.parametrize("answer", [[], None])
def test_empty_or_cancelled_selection_preserves_existing_archive(
    tmp_path, export_run, monkeypatch, answer
):
    import questionary

    run = export_run("model")
    output = tmp_path / "models.zip"
    output.write_bytes(b"previous archive")
    monkeypatch.setattr(
        questionary, "checkbox", lambda *args, **kwargs: SimpleNamespace(ask=lambda: answer)
    )
    result = CliRunner().invoke(export_models, ["--run", str(run), "--output", str(output)])
    assert result.exit_code == (1 if answer is None else 0)
    assert output.read_bytes() == b"previous archive"


def test_inference_checkpoint_fallback_does_not_hide_a_missing_recorded_best(tmp_path, export_run):
    from birdclef.models import FitState

    run = export_run("paused")
    with pytest.raises(ValueError, match="best checkpoint"):
        FitState().best_path(run)
    assert FitState().inference_path(run) == run / "checkpoints/last.ckpt"
    with pytest.raises(FileNotFoundError):
        FitState(best_checkpoint="checkpoints/missing.ckpt").inference_path(run)
    outside = tmp_path / "outside.ckpt"
    outside.touch()
    with pytest.raises(ValueError, match="stay inside"):
        FitState(best_checkpoint="../outside.ckpt").inference_path(run)
