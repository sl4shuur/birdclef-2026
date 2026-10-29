import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from birdclef.data.sequences import (  # noqa: E402
    recording_windows,
    requested_indices,
    sequence_dataset,
    window_targets,
)


def test_context_expands_recordings_but_missing_labels_are_masked(audio_frames):
    _, valid = audio_frames
    frame = pd.concat([valid, valid.assign(filename="other.wav", path="other.wav")]).iloc[::-1]
    full = recording_windows(frame)
    assert len(full) == 24
    assert full.groupby("filename").start.apply(list).tolist() == [list(range(0, 60, 5))] * 2
    targets, mask = window_targets(full, frame, ["a", "b"])
    assert mask.sum() == 4
    assert targets[~mask].eq(0).all()
    dataset = sequence_dataset(torch.arange(48).reshape(24, 2), targets, mask, length=12)
    assert dataset[0][0][:, 0].tolist() == list(range(0, 24, 2))
    assert dataset[1][0][:, 0].tolist() == list(range(24, 48, 2))
    assert requested_indices(frame, full).tolist() == [1, 0, 13, 12]


def test_focal_padding_is_not_stored_or_supervised():
    features = torch.randn(3, 7)
    dataset = sequence_dataset(
        features, torch.ones(3, 2), torch.ones(3, dtype=torch.bool), length=1
    )
    assert dataset.tensors[0].shape == (3, 1, 7)
    inputs, targets, mask, length = dataset[0]
    assert inputs.shape == (12, 7) and targets.shape == (12, 2)
    assert length == 1 and mask.sum() == 1 and mask[0]
    torch.testing.assert_close(inputs[0], features[0])


def test_rejects_duplicate_and_out_of_range_windows(audio_frames):
    _, valid = audio_frames
    with pytest.raises(ValueError, match="unique"):
        recording_windows(pd.concat([valid, valid]))
    with pytest.raises(ValueError, match="aligned"):
        recording_windows(valid.iloc[:1].assign(start=-5))
