import json
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from birdclef.config import DataConfig, LRSchedulerConfig, TrainConfig  # noqa: E402
from birdclef.data.embeddings import embeddings  # noqa: E402
from birdclef.data.prepared import audio_training_datasets  # noqa: E402
from birdclef.data.sampling import MixedSourceDataset  # noqa: E402
from birdclef.models.classifier.backends import EfficientNetBackend  # noqa: E402
from birdclef.models.config import AudioExperimentConfig, AudioModelConfig  # noqa: E402
from birdclef.models.distillation.config import DistillationConfig  # noqa: E402
from birdclef.models.distillation.fit import distillation_data  # noqa: E402
from birdclef.models.distillation.network import FeatureDistillationModule  # noqa: E402
from birdclef.models.encoders.efficientnet import EfficientNetEncoder  # noqa: E402
from birdclef.models.encoders.perch import PerchEncoder  # noqa: E402
from birdclef.processing import log_mel  # noqa: E402


def test_cached_targets_match_centered_crops_and_never_extract(
    tmp_path, encoder_path, audio_frames
):
    train, valid = audio_frames
    config = DistillationConfig(teacher_path=encoder_path, data=DataConfig(cache_dir=tmp_path))
    teacher = PerchEncoder(encoder_path)
    audio, validation = audio_training_datasets(
        train, valid, classes=["a", "b"], transform=None, random_crop=False
    )
    cache = tmp_path / "perch-center-v1" / teacher.sha256
    for source, dataset in audio.by_source.items():
        embeddings(
            train.loc[train.source.eq(source)],
            dataset,
            teacher,
            loader=config.loader,
            cache_dir=cache,
        )
    embeddings(valid, validation, teacher, loader=config.loader, cache_dir=cache)
    with patch.object(PerchEncoder, "__call__", side_effect=AssertionError("cache miss")):
        data = distillation_data(config, train, valid, ["a", "b"], teacher)
    assert isinstance(data.train, MixedSourceDataset)
    waveform, _ = audio.by_source["focal"][0]
    mel, target = data.train.by_source["focal"][0]
    torch.testing.assert_close(mel, log_mel(waveform), rtol=0, atol=0)
    torch.testing.assert_close(target, teacher(waveform.unsqueeze(0))[0])
    torch.testing.assert_close(data.train.by_source["focal"][0][0], mel, rtol=0, atol=0)
    with pytest.raises(ValueError, match="overlap"):
        distillation_data(
            config, train, train.loc[train.source.eq("soundscape")], ["a", "b"], teacher
        )
    next(cache.glob("*.npy")).unlink()
    with pytest.raises(ValueError, match="missing"):
        distillation_data(config, train, valid, ["a", "b"], teacher)


def test_cosine_loss_trains_encoder_and_dynamic_projection():
    encoder = torch.nn.Linear(4, 3)
    module = FeatureDistillationModule(encoder, 3, 7, AudioModelConfig(), LRSchedulerConfig())
    inputs, targets = torch.randn(5, 4), torch.randn(5, 7)
    assert module(inputs).shape == targets.shape
    loss = module.loss((inputs, targets))
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in [encoder.weight, module.projection.weight]:
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    torch.testing.assert_close(loss, module.loss((inputs, targets * 10)))


def test_student_weights_initialize_encoder_without_pretrained_download(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    original = EfficientNetEncoder(pretrained=False)
    path = tmp_path / "encoder.pt"
    torch.save(original.state_dict(), path)
    config = AudioExperimentConfig(model=AudioModelConfig(encoder_weights=path))
    import birdclef.models.encoders.efficientnet as network

    build = network.efficientnet_b0

    def offline_backbone(*, weights):
        assert weights is None
        return build(weights=None)

    monkeypatch.setattr(network, "efficientnet_b0", offline_backbone)
    backend = EfficientNetBackend(config, ["a", "b"])
    restored = backend.network_encoder(initialize_pretrained=True)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    path.unlink()
    # Inference and checkpoint resume do not depend on the Stage-1 file.
    backend.network_encoder(initialize_pretrained=False)
    assert "encoder_weights" in json.loads(config.model_dump_json())["model"]


def test_distillation_rejects_classifier_monitor_and_partial_freezing(encoder_path):
    with pytest.raises(ValueError, match="val/loss"):
        DistillationConfig(
            teacher_path=encoder_path,
            train=TrainConfig(checkpoint_metric="val/auc", checkpoint_mode="max"),
        )
    with pytest.raises(ValueError, match="entire student"):
        DistillationConfig(teacher_path=encoder_path, model=AudioModelConfig(unfreeze_last_n=2))
