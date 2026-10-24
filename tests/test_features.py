import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from birdclef.data.features import FeatureDataset, multilabel_dataset  # noqa: E402


def test_dataset_preserves_row_feature_and_target_order():
    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4], "bird": [0, 1], "owl": [1, 0]}, index=[9, 2])
    original = frame.copy()
    dataset = multilabel_dataset(frame, feature_columns=["y", "x"], target_columns=["owl", "bird"])
    inputs, targets = dataset[0]
    assert inputs.dtype == targets.dtype == torch.float32
    assert inputs.tolist() == [3, 1]
    assert targets.tolist() == [1, 0]
    assert dataset[1][1].tolist() == [0, 1]
    inference = FeatureDataset(frame[["x", "y"]], ["y", "x"])
    batches = list(torch.utils.data.DataLoader(inference, batch_size=2))
    assert len(inference) == 2
    assert isinstance(batches[0], torch.Tensor)
    assert batches[0].tolist() == [[3, 1], [4, 2]]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("feature, target", [(float("nan"), 0), (1, float("inf")), (1, 0.5)])
def test_dataset_rejects_nonfinite_values_and_nonbinary_targets(feature, target):
    with pytest.raises(ValueError):
        multilabel_dataset(
            pd.DataFrame({"feature": [feature], "bird": [target]}),
            feature_columns=["feature"],
            target_columns=["bird"],
        )
