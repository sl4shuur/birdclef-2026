import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

from birdclef.data.audio import (  # noqa: E402
    FocalAudioDataset,
    SoundscapeDataset,
    read_audio_window,
)
from birdclef.data.manifests import (  # noqa: E402
    filter_focal_duration,
    focal_manifest,
    soundscape_manifest,
)
from birdclef.data.split import group_folds  # noqa: E402
from birdclef.processing.audio import log_mel  # noqa: E402


def test_csv_round_trip_aligns_labels_by_identity_and_rejects_lost_keys(tmp_path):
    for name in ["a.wav", "b.wav"]:
        (tmp_path / name).touch()
    frame = pd.DataFrame(
        {
            "filename": ["a.wav", "b.wav"],
            "primary_label": ["a", "b"],
            "secondary_labels": ["[]", "[]"],
        }
    )
    metadata, targets = focal_manifest(frame, tmp_path, ["b", "a"])
    targets.to_csv(tmp_path / "targets.csv")
    restored = pd.read_csv(tmp_path / "targets.csv", index_col="filename")
    reordered = metadata.iloc[::-1].reset_index(drop=True)
    dataset = FocalAudioDataset(reordered, restored, training=False)
    assert dataset.targets.tolist() == [[1, 0], [0, 1]]
    for bad in [
        restored.reset_index(drop=True),
        restored.iloc[:1],
        pd.concat([restored, restored.iloc[:1]]),
    ]:
        with pytest.raises(ValueError, match="keys"):
            FocalAudioDataset(reordered, bad, training=False)


def test_duration_policy_excludes_below_one_second_but_keeps_boundary(tmp_path):
    frame = pd.DataFrame(
        {
            "filename": ["tiny.wav", "one.wav"],
            "primary_label": ["a", "b"],
            "secondary_labels": ["[]", "[]"],
        }
    )
    for filename in frame.filename:
        (tmp_path / filename).touch()
    metadata, targets = focal_manifest(frame, tmp_path, ["a", "b"])
    headers = pd.DataFrame({"filename": ["one.wav", "tiny.wav"], "duration": [1.0, 0.999]})
    retained, labels, excluded = filter_focal_duration(metadata, targets, headers)
    assert retained.filename.tolist() == ["one.wav"]
    assert labels.index.tolist() == ["one.wav"]
    assert labels.iloc[0].tolist() == [0, 1]
    assert excluded.filename.tolist() == ["tiny.wav"]
    assert (tmp_path / "tiny.wav").exists()


def test_window_targets_deduplicate_and_never_fill_missing_windows(tmp_path):
    path = tmp_path / "sc.wav"
    sf.write(path, np.zeros(16000 * 10, dtype=np.float32), 16000)
    row = {"filename": "sc.wav", "start": "00:00:05", "end": "00:00:10", "primary_label": "a;b"}
    annotations = pd.DataFrame([row, dict(row, primary_label="b;a")])
    metadata, targets = soundscape_manifest(annotations, tmp_path, ["b", "a"])
    assert len(metadata) == 1
    assert targets.index.tolist() == ["sc_10"]
    assert targets.iloc[0].tolist() == [1, 1]
    inputs, labels = SoundscapeDataset(metadata, targets)[0]
    assert inputs.shape == (160000,)
    assert labels.tolist() == [1, 1]
    for bad in [
        pd.DataFrame([row, dict(row, primary_label="a")]),
        pd.DataFrame([dict(row, start="00:00:01")]),
        pd.DataFrame([dict(row, primary_label=None)]),
        pd.DataFrame([dict(row, primary_label="unknown")]),
    ]:
        with pytest.raises(ValueError):
            soundscape_manifest(bad, tmp_path, ["b", "a"])


def test_decode_downmix_resample_padding_and_strict_truncation(tmp_path):
    source_rate = 16000
    signal = np.full(source_rate, 0.25, dtype=np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([signal, signal * 3]), source_rate, subtype="FLOAT")
    waveform = read_audio_window(path, pad=True)
    assert waveform.dtype == torch.float32
    assert waveform.shape == (160000,)
    assert torch.allclose(waveform[100:31000], torch.full((30900,), 0.5), atol=0.001)
    assert torch.count_nonzero(waveform[32000:]) == 0
    with pytest.raises(ValueError, match="Truncated"):
        read_audio_window(path)
    mel = log_mel(waveform)
    assert mel.shape == (1, 128, 497)
    assert torch.isfinite(mel).all()


def test_focal_crops_reproduce_with_seed_and_eval_is_deterministic(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(path, np.linspace(-0.5, 0.5, 8 * 32000, dtype=np.float32), 32000)
    metadata = pd.DataFrame({"filename": [path.name], "path": [str(path)]})
    targets = pd.DataFrame({"a": [1]}, index=pd.Index([path.name], name="filename"))
    train = FocalAudioDataset(metadata, targets, training=True)
    torch.manual_seed(4)
    first = train[0][0]
    torch.manual_seed(4)
    assert torch.equal(first, train[0][0])
    valid = FocalAudioDataset(metadata, targets, training=False)
    assert torch.equal(valid[0][0], valid[0][0])
    expected = read_audio_window(path, start=1.5)
    assert torch.equal(valid[0][0], expected)


def test_folds_are_disjoint_and_stable_when_rows_are_reordered():
    frame = pd.DataFrame({"filename": ["a", "a", "b", "c", "d", "e"]})
    folds = group_folds(frame)
    assert folds.groupby(frame.filename).nunique().eq(1).all()
    assert set(folds) == set(range(5))
    pd.testing.assert_series_equal(
        folds, group_folds(frame.sample(frac=1, random_state=7)).sort_index()
    )
