import json
import shutil
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pandas as pd
import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from lightning.pytorch.loggers import CSVLogger  # noqa: E402

from birdclef.config import (  # noqa: E402
    LoaderConfig,
    LRSchedulerConfig,
    RuntimeConfig,
    TrainConfig,
)
from birdclef.data.sequences import (  # noqa: E402
    recording_windows,
)
from birdclef.experiment import RunContext  # noqa: E402
from birdclef.models import (  # noqa: E402
    RecurrentClassifier,
    RecurrentExperimentConfig,
    RecurrentModelConfig,
)
from birdclef.models.recurrent.network import RecurrentClassifierModule, RecurrentHead  # noqa: E402
from birdclef.models.runs import read_model_config  # noqa: E402
from birdclef.submission import predict_runs  # noqa: E402
from birdclef.submission.export import main as export_models  # noqa: E402


def model_config(**kwargs):
    return RecurrentModelConfig(
        encoder_path=Path("test.onnx"), recurrent_dim=3, dropout=0, **kwargs
    )


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_packed_focal_predictions_ignore_the_padded_tail(cell):
    torch.manual_seed(2)
    head = RecurrentHead(7, 4, model_config(cell=cell)).eval()
    first = torch.randn(1, 1, 7)
    padded = torch.cat([first, torch.randn(1, 11, 7) * 100], dim=1)
    expected = head(first)
    actual = head(padded, torch.tensor([1]))
    assert actual.shape == (1, 12, 4)
    torch.testing.assert_close(actual[:, :1], expected)


def test_masked_loss_ignores_missing_targets_but_uses_future_context():
    torch.manual_seed(42)
    module = RecurrentClassifierModule(7, model_config(), LRSchedulerConfig(), classes=["a", "b"])
    inputs = torch.randn(2, 12, 7, requires_grad=True)
    targets = torch.zeros(2, 12, 2)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, 0] = True
    lengths = torch.tensor([12, 1])
    loss = module.loss((inputs, targets, mask, lengths))
    changed = targets.masked_fill(~mask.unsqueeze(-1), float("nan"))
    torch.testing.assert_close(loss, module.loss((inputs, changed, mask, lengths)))
    loss.backward()
    assert inputs.grad is not None
    assert inputs.grad[0, -1].abs().sum() > 0
    assert inputs.grad[1, 1:].eq(0).all()


def test_smoke_sampling_preserves_all_annotations_of_selected_recordings(audio_frames):
    train, _ = audio_frames
    focal = train.loc[train.source.eq("focal")].assign(row_id=pd.NA)
    records = [
        train.loc[train.source.eq("soundscape")].assign(filename=f"clip-{i}.wav") for i in range(40)
    ]
    config = RecurrentExperimentConfig(model=model_config())
    model = RecurrentClassifier(config, classes=["a", "b"])
    sampled = model.smoke_training_data(pd.concat([focal, *records], ignore_index=True))
    soundscapes = sampled.loc[sampled.source.eq("soundscape")]
    assert soundscapes.filename.nunique() == 32
    assert soundscapes.groupby("filename").size().eq(2).all()
    assert sampled.source.eq("focal").sum() == len(focal)


@pytest.mark.slow
def test_recurrent_train_resume_export_and_uncached_cpu_inference(
    tmp_path, encoder_path, audio_frames
):
    train, valid = audio_frames
    config = RecurrentExperimentConfig(
        model=RecurrentModelConfig(encoder_path=encoder_path, recurrent_dim=3, dropout=0),
        runtime=RuntimeConfig(accelerator="cpu", devices=1, precision="32-true"),
        train=TrainConfig(max_epochs=1, checkpoint_metric="val/auc", checkpoint_mode="max"),
        loader=LoaderConfig(batch_size=2, predict_batch_size=2, val_batch_size=1),
    )
    config.data.cache_dir = tmp_path / "cache"
    run = tmp_path / "run"
    context = RunContext(run, CSVLogger(tmp_path, name="logs"), Mock(), "sequence-fold")
    model = RecurrentClassifier(config, classes=["a", "b"])
    model.fit(train, valid, context)
    metrics = json.loads((run / "validation-metrics.json").read_text())
    assert metrics["windows"] == len(valid) and metrics["auc_classes"] == 2
    assert len(list(config.data.cache_dir.rglob("*.npy"))) == 25
    expected = model.predict(valid)
    full = recording_windows(valid)
    context_prediction = model.predict(full)
    pd.testing.assert_frame_equal(context_prediction.iloc[:2], expected)
    config.train.max_epochs = 2
    model.fit(train, valid, context)
    assert json.loads((run / "fit.json").read_text())["trained_epochs"] == 2
    expected = model.predict(valid)
    assert isinstance(read_model_config(run / "model-config.json"), RecurrentExperimentConfig)
    archive = tmp_path / "models.zip"
    result = CliRunner().invoke(
        export_models, ["--run", str(run), "--all", "--output", str(archive)]
    )
    assert result.exit_code == 0, result.output
    with ZipFile(archive) as packed:
        packed.extractall(tmp_path / "unpacked")
    encoder_path.unlink()
    shutil.rmtree(config.data.cache_dir)
    actual = predict_runs([tmp_path / "unpacked/run"], valid.iloc[::-1], expected)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
