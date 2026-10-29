import json
import shutil
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
sf = pytest.importorskip("soundfile")

from lightning.pytorch.loggers import CSVLogger  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from birdclef.config import LoaderConfig, RuntimeConfig, TrainConfig  # noqa: E402
from birdclef.data import InferenceSoundscapeDataset  # noqa: E402
from birdclef.data.embeddings import embeddings  # noqa: E402
from birdclef.experiment import RunContext  # noqa: E402
from birdclef.models import AudioClassifier, AudioExperimentConfig, AudioModelConfig  # noqa: E402
from birdclef.models.encoders.config import EncoderName  # noqa: E402
from birdclef.models.encoders.perch import PerchEncoder, file_sha256  # noqa: E402
from birdclef.submission import predict_runs  # noqa: E402
from birdclef.submission.export import main as export_models  # noqa: E402


@pytest.mark.parametrize("fixed_batch", [None, 1, 2])
def test_encoder_respects_export_batch_size_including_the_last_partial_batch(
    encoder_path, fixed_batch
):
    if fixed_batch:
        graph = onnx.load(encoder_path)
        for value in [graph.graph.input[0], graph.graph.output[0]]:
            value.type.tensor_type.shape.dim[0].dim_value = fixed_batch
        onnx.save(graph, encoder_path)
    encoder = PerchEncoder(encoder_path)
    inputs = torch.stack([torch.full((160000,), value) for value in (0.1, 0.2, 0.3)])
    output = encoder(inputs)
    np.testing.assert_allclose(output, [[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]], atol=1e-4)
    assert not output.requires_grad
    with pytest.raises(ValueError, match="160000"):
        encoder(torch.zeros(1, 100))


def test_cache_reuses_readonly_entries_after_relocation_and_extracts_only_misses(
    tmp_path, encoder_path, audio_frames, monkeypatch
):
    _, frame = audio_frames
    encoder = PerchEncoder(encoder_path)
    loader = LoaderConfig(predict_batch_size=2)
    cache = tmp_path / "cache" / encoder.sha256
    expected = embeddings(
        frame, InferenceSoundscapeDataset(frame), encoder, loader=loader, cache_dir=cache
    )
    relocated = frame.iloc[::-1].assign(path=str(tmp_path / "missing.wav"), fold=99, a=0, b=1)

    class NoAudioReads(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            raise AssertionError("A complete cache must not decode audio.")

    fresh = PerchEncoder(encoder_path)
    with monkeypatch.context() as patch:
        patch.setattr(fresh, "_load", Mock(side_effect=AssertionError("Cache must skip ONNX.")))
        patch.setattr(Path, "mkdir", Mock(side_effect=AssertionError("Cache is read-only.")))
        actual = embeddings(
            relocated, NoAudioReads(), fresh, loader=loader, cache_dir=cache, cache_only=True
        )
    torch.testing.assert_close(actual, expected.flip(0))
    next(cache.rglob("*.npy")).unlink()
    with pytest.raises(ValueError, match="prepare embeddings in one process"):
        embeddings(
            relocated, NoAudioReads(), fresh, loader=loader, cache_dir=cache, cache_only=True
        )
    calls = []
    original = PerchEncoder.__call__

    def counted(self, inputs):
        calls.append(len(inputs))
        return original(self, inputs)

    monkeypatch.setattr(PerchEncoder, "__call__", counted)
    actual = embeddings(
        frame, InferenceSoundscapeDataset(frame), fresh, loader=loader, cache_dir=cache
    )
    torch.testing.assert_close(actual, expected)
    assert calls == [1]
    graph = onnx.load(encoder_path)
    graph.producer_name = "different-encoder"
    onnx.save(graph, encoder_path)
    embeddings(
        frame,
        InferenceSoundscapeDataset(frame),
        PerchEncoder(encoder_path),
        loader=loader,
        cache_dir=cache.parent / file_sha256(encoder_path),
    )
    assert calls == [1, 2]


@pytest.mark.parametrize("rank", ["0", "1"])
def test_ddp_workers_require_an_indexed_cache_before_extraction(
    tmp_path, encoder_path, audio_frames, monkeypatch, rank
):
    from birdclef.models.classifier.backends import PerchBackend

    train, valid = audio_frames
    config = AudioExperimentConfig(
        model=AudioModelConfig(
            encoder=EncoderName.PERCH, encoder_path=encoder_path, unfreeze_epoch=None
        )
    )
    config.data.cache_dir = tmp_path / "cache"
    context = RunContext(tmp_path / "run", Mock(), Mock(), "test-fold")
    monkeypatch.setenv("LOCAL_RANK", rank)
    monkeypatch.setattr(
        PerchEncoder, "_load", Mock(side_effect=AssertionError("DDP must not extract features."))
    )
    with pytest.raises(ValueError, match="prepare embeddings in one process"):
        PerchBackend(config, ["a", "b"]).training_data(train, valid, context)
    assert not config.data.cache_dir.exists()


@pytest.mark.slow
@pytest.mark.parametrize("recurrent", [False, True])
def test_script_ddp_prepares_cache_once_before_workers(
    tmp_path, encoder_path, audio_frames, recurrent
):
    import os
    import subprocess
    import sys

    train, valid = audio_frames
    train.to_csv(tmp_path / "train.csv", index=False)
    valid.to_csv(tmp_path / "valid.csv", index=False)
    worker = Path(__file__).with_name("ddp_perch_worker.py")
    result = subprocess.run(
        [
            sys.executable,
            str(worker),
            str(tmp_path),
            str(encoder_path),
            *(["recurrent"] if recurrent else []),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = 25 if recurrent else len(train) + len(valid)
    assert sum(map(int, (tmp_path / "extracted.txt").read_text().splitlines())) == expected
    assert json.loads((tmp_path / "run/validation-metrics.json").read_text())["windows"] == len(
        valid
    )
    assert {path.name for path in tmp_path.glob("trained-rank-*")} == {
        "trained-rank-0",
        "trained-rank-1",
    }
    metrics = json.loads((tmp_path / "run/validation-metrics.json").read_text())
    state = json.loads((tmp_path / "run/fit.json").read_text())
    checkpoint = torch.load(
        tmp_path / "run" / state["best_checkpoint"], map_location="cpu", weights_only=False
    )
    best = next(
        value["best_model_score"].item()
        for key, value in checkpoint["callbacks"].items()
        if "val/auc" in key and value.get("best_model_score") is not None
    )
    assert abs(best - metrics["auc"]) < 1e-6


@pytest.mark.slow
def test_cached_mlp_trains_resumes_and_exports_portable_offline_predictions(
    tmp_path, encoder_path, audio_frames
):
    train, valid = audio_frames
    config = AudioExperimentConfig(
        model=AudioModelConfig(
            encoder=EncoderName.PERCH, encoder_path=encoder_path, unfreeze_epoch=None, hidden_dim=4
        ),
        train=TrainConfig(max_epochs=1),
        runtime=RuntimeConfig(accelerator="cpu", devices=1, precision="32-true"),
        loader=LoaderConfig(batch_size=2, predict_batch_size=2, val_batch_size=1),
    )
    config.data.cache_dir = tmp_path / "cache"
    run = tmp_path / "run"
    context = RunContext(run, CSVLogger(tmp_path, name="logs"), Mock(), "test-fold")
    encoder_hash = file_sha256(encoder_path)
    model = AudioClassifier(config, classes=["a", "b"])
    model.fit(train, valid, context)
    assert model._module is not None
    head = model._module.head
    assert isinstance(head, nn.Sequential)
    assert head[1].in_features == 2
    assert head[-1].out_features == 2
    assert file_sha256(encoder_path) == encoder_hash
    assert json.loads((run / "validation-metrics.json").read_text())["auc_classes"] == 2
    config.train.max_epochs = 2
    model.fit(train, valid, context)
    assert json.loads((run / "fit.json").read_text())["trained_epochs"] == 2
    expected = model.predict(valid)
    # A GPU extraction setting must not require CUDA in the exported CPU submission.
    saved_path = run / "model-config.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    saved["model"]["encoder_device"] = "cuda"
    saved_path.write_text(json.dumps(saved), encoding="utf-8")
    archive = tmp_path / "export.zip"
    exported = CliRunner().invoke(
        export_models, ["--run", str(run), "--all", "--output", str(archive)]
    )
    assert exported.exit_code == 0, exported.output
    with ZipFile(archive) as packed:
        packed.extractall(tmp_path / "unpacked")
    encoder_path.unlink()
    shutil.rmtree(config.data.cache_dir)
    restored = AudioClassifier.from_run(tmp_path / "unpacked/run")
    actual = restored.predict(valid.iloc[::-1])
    pd.testing.assert_frame_equal(actual, expected.iloc[::-1].reset_index(drop=True))
    ensemble = predict_runs([tmp_path / "unpacked/run"], valid, expected)
    pd.testing.assert_frame_equal(ensemble, expected, check_dtype=False)
    (tmp_path / "unpacked/run/classes.json").write_text(json.dumps(["b", "a"]))
    with pytest.raises(ValueError, match="class order"):
        AudioClassifier.from_run(tmp_path / "unpacked/run").predict(valid)


def test_perch_rejects_unfreezing_and_missing_weights_path():
    with pytest.raises(ValueError, match="encoder_path"):
        AudioModelConfig(encoder=EncoderName.PERCH, unfreeze_epoch=None)
    with pytest.raises(ValueError, match="frozen"):
        AudioModelConfig(
            encoder=EncoderName.PERCH, encoder_path=Path("perch.onnx"), unfreeze_epoch=0
        )


def test_cuda_request_fails_before_silently_using_cpu(encoder_path, monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        PerchEncoder(encoder_path, device="cuda")(torch.zeros(1, 160000))
