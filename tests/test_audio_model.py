import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
lightning = pytest.importorskip("lightning")
sf = pytest.importorskip("soundfile")

from birdclef.config import LoaderConfig, LRSchedulerConfig, RuntimeConfig  # noqa: E402
from birdclef.data.audio import read_audio_window  # noqa: E402
from birdclef.data.prepared import audio_training_datasets, load_audio_split  # noqa: E402
from birdclef.models.audio_cnn import network  # noqa: E402
from birdclef.models.audio_cnn.config import AudioExperimentConfig, AudioModelConfig  # noqa: E402
from birdclef.models.audio_cnn.model import AudioCNN  # noqa: E402
from birdclef.models.audio_cnn.network import AudioCNNModule  # noqa: E402
from birdclef.processing.audio import log_mel  # noqa: E402


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_network_shapes_and_gradients_reach_encoder_and_head():
    module = AudioCNNModule(
        AudioModelConfig(target_columns=["a", "b"], pretrained=False), LRSchedulerConfig()
    )
    inputs = torch.randn(2, 1, 128, 497)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = module(inputs)
    assert logits.shape == (2, 2)
    loss = module.loss((inputs, targets))
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in [module.encoder[0][0].weight, module.head[1].weight]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert isinstance(module.configure_optimizers(), torch.optim.AdamW)
    assert all(parameter.requires_grad for parameter in module.parameters())


def test_pretrained_stem_averages_rgb_and_offline_initialization_skips_weights(monkeypatch):
    original_factory = network.efficientnet_b0
    seen = []

    def fake_factory(*, weights):
        seen.append(weights)
        backbone = original_factory(weights=None)
        with torch.no_grad():
            for channel, value in enumerate([1.0, 2.0, 3.0]):
                backbone.features[0][0].weight[:, channel].fill_(value)
        return backbone

    monkeypatch.setattr(network, "efficientnet_b0", fake_factory)
    config = AudioModelConfig(target_columns=["a"], pretrained=True)
    module = AudioCNNModule(config, LRSchedulerConfig())
    assert seen[-1] == network.EfficientNet_B0_Weights.IMAGENET1K_V1
    assert module.encoder[0][0].weight.shape[1] == 1
    assert module.encoder[0][0].weight.eq(2).all()
    AudioCNNModule(config, LRSchedulerConfig(), initialize_pretrained=False)
    assert seen[-1] is None


def test_validation_is_pooled_and_checkpoint_restores_predictions_without_download(
    tmp_path, monkeypatch
):
    original_factory = network.efficientnet_b0
    seen = []

    def offline_factory(*, weights):
        seen.append(weights)
        assert weights is None
        return original_factory(weights=None)

    monkeypatch.setattr(network, "efficientnet_b0", offline_factory)
    config = AudioExperimentConfig(
        model=AudioModelConfig(target_columns=["a", "b"], pretrained=True),
        runtime=RuntimeConfig(accelerator="cpu", devices=1, precision="32-true"),
        loader=LoaderConfig(predict_batch_size=2),
    )
    module = AudioCNNModule(config.model, config.lr_scheduler, initialize_pretrained=False)
    # Each separate batch has no negatives for one species; the full fold is valid.
    inputs = torch.randn(4, 1, 128, 497)
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets), batch_size=2
    )
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert torch.isfinite(trainer.callback_metrics["val/auc"])
    assert trainer.callback_metrics["val/auc_classes"].item() == 2
    trainer.save_checkpoint(tmp_path / "best.ckpt")
    (tmp_path / "model-config.json").write_text(json.dumps(config.safe_dump()), encoding="utf-8")
    (tmp_path / "fit.json").write_text(
        json.dumps(
            {"trained_epochs": 1, "stop_reason": "max_epochs", "best_checkpoint": "best.ckpt"}
        ),
        encoding="utf-8",
    )
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(5 * 32000, dtype=np.float32), 32000)
    manifest = pd.DataFrame(
        {
            "row_id": ["clip_5"],
            "filename": ["clip.wav"],
            "path": [str(path)],
            "start": [0],
            "end": [5],
        }
    )
    module.eval()
    with torch.inference_mode():
        expected = module.predict_batch(log_mel(read_audio_window(path)).unsqueeze(0))
    restored = AudioCNN.from_run(tmp_path)
    result = restored.predict(manifest)
    assert result.columns.tolist() == ["row_id", "a", "b"]
    assert result.row_id.tolist() == ["clip_5"]
    np.testing.assert_allclose(result[["a", "b"]], expected.numpy(), atol=1e-6)
    assert all(weights is None for weights in seen)


def test_prepared_split_checks_identity_and_preserves_recording_groups(tmp_path):
    classes = ["a", "b"]
    (tmp_path / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    pd.DataFrame({"filename": ["focal.wav"]}).to_csv(tmp_path / "focal.csv", index=False)
    pd.DataFrame({"a": [1], "b": [0]}, index=pd.Index(["focal.wav"], name="filename")).to_csv(
        tmp_path / "focal_targets.csv"
    )
    metadata = pd.DataFrame(
        {
            "filename": ["x.wav", "x.wav", "y.wav", "y.wav"],
            "row_id": ["x_5", "x_10", "y_5", "y_10"],
            "start": [0, 5, 0, 5],
            "end": [5, 10, 5, 10],
            "fold": [0, 0, 1, 1],
        }
    )
    metadata.to_csv(tmp_path / "soundscapes.csv", index=False)
    pd.DataFrame(
        {"a": [1, 0, 1, 0], "b": [0, 1, 0, 1]}, index=pd.Index(metadata.row_id, name="row_id")
    ).to_csv(tmp_path / "soundscape_targets.csv")
    names = [
        "classes.json",
        "focal.csv",
        "focal_targets.csv",
        "soundscapes.csv",
        "soundscape_targets.csv",
    ]
    manifest = {
        "version": 2,
        "artifacts_sha256": {
            name: sha256((tmp_path / name).read_bytes()).hexdigest() for name in names
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    train, valid = load_audio_split(tmp_path, tmp_path, fold=0, classes=classes)
    assert valid.row_id.tolist() == ["x_5", "x_10"]
    assert train.loc[train.source.eq("soundscape"), "filename"].tolist() == ["y.wav", "y.wav"]
    training, validation = audio_training_datasets(train, valid, classes=classes)
    assert len(training) == 3 and len(validation) == 2
    leaking = train.copy()
    leaking.loc[leaking.source.eq("soundscape"), "filename"] = "x.wav"
    with pytest.raises(ValueError, match="overlap"):
        audio_training_datasets(leaking, valid, classes=classes)
    (tmp_path / "classes.json").write_text(json.dumps(classes[::-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_audio_split(tmp_path, tmp_path, fold=0, classes=classes)
