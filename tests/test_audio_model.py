import json
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
lightning = pytest.importorskip("lightning")
sf = pytest.importorskip("soundfile")

from torch import nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from birdclef.config import LoaderConfig, LRSchedulerConfig, RuntimeConfig  # noqa: E402
from birdclef.data.audio import read_audio_window  # noqa: E402
from birdclef.data.prepared import audio_training_datasets, load_audio_split  # noqa: E402
from birdclef.models.audio_cnn import network  # noqa: E402
from birdclef.models.audio_cnn.config import AudioExperimentConfig, AudioModelConfig  # noqa: E402
from birdclef.models.audio_cnn.model import AudioCNN  # noqa: E402
from birdclef.models.audio_cnn.network import AudioCNNModule  # noqa: E402
from birdclef.processing.audio import log_mel  # noqa: E402


def first_conv(features: nn.Sequential) -> nn.Conv2d:
    stem = features[0]
    assert isinstance(stem, nn.Sequential)
    convolution = stem[0]
    assert isinstance(convolution, nn.Conv2d)
    return convolution


def test_two_process_training_unfreezes_synchronously_and_scores_without_padding(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    worker = Path(__file__).with_name("ddp_audio_worker.py")
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "WANDB_MODE": "offline",
    }
    result = subprocess.run(
        [sys.executable, str(worker), str(tmp_path), "cpu"],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "ddp-check.json").read_text())["status"] == "passed"


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_network_shapes_and_gradients_reach_encoder_and_head():
    module = AudioCNNModule(
        AudioModelConfig(pretrained=False), LRSchedulerConfig(), classes=["a", "b"]
    )
    inputs = torch.randn(2, 1, 128, 497)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = module(inputs)
    assert logits.shape == (2, 2)
    loss = module.loss((inputs, targets))
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in [first_conv(module.encoder).weight, module.head[1].weight]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert isinstance(module.configure_optimizers(), torch.optim.AdamW)
    assert all(parameter.requires_grad for parameter in module.parameters())


@pytest.mark.parametrize("last_n", [None, 2])
def test_frozen_blocks_keep_weights_and_batchnorm_statistics_while_head_trains(last_n):
    config = AudioModelConfig(
        pretrained=False,
        unfreeze_epoch=None if last_n is None else 0,
        unfreeze_last_n=last_n,
    )
    module = AudioCNNModule(config, LRSchedulerConfig(), classes=["a", "b"])
    module.eval()
    module.train()  # Lightning switches back to training after validation.
    boundary = len(module.encoder) - (last_n or 0)
    before = [
        {key: value.clone() for key, value in block.state_dict().items()}
        for block in module.encoder
    ]
    head = module.head[1]
    assert isinstance(head, nn.Linear)
    head_before = head.weight.detach().clone()
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, AdamW)
    assert [group["name"] for group in optimizer.param_groups] == ["encoder", "head"]
    assert [group["lr"] for group in optimizer.param_groups] == [config.encoder_lr, config.head_lr]
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        list(module.parameters())
    )
    module.loss((torch.randn(2, 1, 128, 497), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))).backward()
    optimizer.step()
    for index, block in enumerate(module.encoder):
        trainable = index >= boundary
        assert block.training == trainable
        assert all(parameter.requires_grad == trainable for parameter in block.parameters())
        after = block.state_dict()
        if trainable:
            assert any(not torch.equal(before[index][key], value) for key, value in after.items())
        else:
            for key, value in after.items():
                torch.testing.assert_close(value, before[index][key], rtol=0, atol=0)
    assert not torch.equal(head.weight, head_before)


@pytest.mark.parametrize("scheduler", ["none", "cosine", "plateau"])
def test_gradual_unfreezing_resumes_at_the_right_epoch_with_optimizer_and_scheduler(
    tmp_path, scheduler
):
    config = AudioModelConfig(pretrained=False, unfreeze_epoch=2)
    schedule = LRSchedulerConfig(name=scheduler, t_max=4)
    module = AudioCNNModule(config, schedule, classes=["a", "b"])
    encoder_before = {key: value.clone() for key, value in module.encoder.state_dict().items()}
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(2, 1, 128, 497), torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ),
        batch_size=2,
    )
    options: dict[str, Any] = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "default_root_dir": tmp_path,
    }
    trainer = lightning.Trainer(max_epochs=2, **options)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    for key, value in module.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder_before[key], rtol=0, atol=0)
    checkpoint = tmp_path / "frozen.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved_lrs = [group["lr"] for group in trainer.optimizers[0].param_groups]
    if scheduler == "cosine":
        assert saved_lrs == pytest.approx([config.encoder_lr / 2, config.head_lr / 2])

    class CheckResume(lightning.Callback):
        def on_train_start(self, trainer, pl_module):
            assert trainer.current_epoch == 2
            assert [group["lr"] for group in trainer.optimizers[0].param_groups] == saved_lrs

    resumed = AudioCNNModule(config, schedule, classes=["a", "b"])
    trainer = lightning.Trainer(max_epochs=3, callbacks=[CheckResume()], **options)
    trainer.fit(
        resumed,
        train_dataloaders=loader,
        val_dataloaders=loader,
        ckpt_path=checkpoint,
        weights_only=False,
    )
    assert trainer.global_step == 3
    assert all(parameter.requires_grad for parameter in resumed.encoder.parameters())
    assert not torch.equal(first_conv(resumed.encoder).weight, encoder_before["0.0.weight"])
    assert first_conv(resumed.encoder).weight in trainer.optimizers[0].state


@pytest.mark.parametrize(
    "settings",
    [
        {"encoder_lr": 0},
        {"head_lr": float("nan")},
        {"unfreeze_epoch": -1},
        {"unfreeze_last_n": 0},
        {"unfreeze_last_n": 10},
    ],
)
def test_rejects_invalid_fine_tuning_settings(settings):
    with pytest.raises(ValueError):
        AudioModelConfig(**settings)


def test_pretrained_stem_averages_rgb_and_offline_initialization_skips_weights(monkeypatch):
    original_factory = network.efficientnet_b0
    seen = []

    def fake_factory(*, weights):
        seen.append(weights)
        backbone = original_factory(weights=None)
        with torch.no_grad():
            for channel, value in enumerate([1.0, 2.0, 3.0]):
                first_conv(backbone.features).weight[:, channel].fill_(value)
        return backbone

    monkeypatch.setattr(network, "efficientnet_b0", fake_factory)
    config = AudioModelConfig(pretrained=True)
    module = AudioCNNModule(config, LRSchedulerConfig(), classes=["a"])
    assert seen[-1] == network.EfficientNet_B0_Weights.IMAGENET1K_V1
    assert first_conv(module.encoder).weight.shape[1] == 1
    assert first_conv(module.encoder).weight.eq(2).all()
    AudioCNNModule(config, LRSchedulerConfig(), classes=["a"], initialize_pretrained=False)
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
        model=AudioModelConfig(pretrained=True),
        runtime=RuntimeConfig(accelerator="cpu", devices=1, precision="32-true"),
        loader=LoaderConfig(predict_batch_size=2),
    )
    module = AudioCNNModule(
        config.model, config.lr_scheduler, classes=["a", "b"], initialize_pretrained=False
    )
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
    (tmp_path / "classes.json").write_text(json.dumps(["a", "b"]))
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
    from birdclef.submission import predict_runs

    ensemble = predict_runs(
        [tmp_path, tmp_path], manifest, result[["row_id", "b", "a"]], weights=[1, 3]
    )
    assert ensemble.columns.tolist() == ["row_id", "b", "a"]
    np.testing.assert_allclose(ensemble[["a", "b"]], expected.numpy(), atol=1e-6)
    assert all(weights is None for weights in seen)
    (tmp_path / "classes.json").write_text(json.dumps(["b", "a"]))
    with pytest.raises(ValueError, match="class order"):
        AudioCNN.from_run(tmp_path).predict(manifest)
    (tmp_path / "classes.json").write_text(json.dumps(["a", "b"]))
    (tmp_path / "classes.json").unlink()
    with pytest.raises(FileNotFoundError, match="classes.json"):
        AudioCNN.from_run(tmp_path)


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


def test_resume_checks_ordered_columns_and_training_policy_in_one_signature(tmp_path):
    from unittest.mock import Mock

    from birdclef.experiment import RunContext
    from birdclef.models import check_fit_inputs

    config = AudioExperimentConfig()
    # Equal target values deliberately make the hashes insensitive to a class swap.
    frame = pd.DataFrame({"filename": ["a"], "bird": [1], "owl": [1]})
    columns = ["filename", "bird", "owl"]
    context = RunContext(tmp_path, Mock(), Mock(), "fold-0")
    check_fit_inputs(config, frame, frame, columns, context)
    path = tmp_path / "fit-inputs.json"
    original = path.read_text(encoding="utf-8")
    assert json.loads(original)["columns"] == columns
    config.train.max_epochs += 1
    config.train.checkpoint_top_k += 1
    check_fit_inputs(config, frame, frame, columns, context)
    with pytest.raises(ValueError, match="settings changed"):
        check_fit_inputs(config, frame, frame, ["filename", "owl", "bird"], context)
    config.model.unfreeze_epoch = 2
    with pytest.raises(ValueError, match="settings changed"):
        check_fit_inputs(config, frame, frame, columns, context)
    assert path.read_text(encoding="utf-8") == original
