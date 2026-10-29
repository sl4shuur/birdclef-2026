import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from birdclef.config import LoaderConfig  # noqa: E402
from birdclef.data.embeddings import embeddings  # noqa: E402
from birdclef.data.features import FeatureDataset  # noqa: E402


def test_extraction_accepts_a_callable_and_preserves_partial_batch_order():
    inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    calls = []

    def extract(batch):
        calls.append(len(batch))
        return batch * 2

    actual = embeddings(
        pd.DataFrame(index=range(5)),
        FeatureDataset(inputs),
        extract,
        loader=LoaderConfig(predict_batch_size=2),
    )
    assert calls == [2, 2, 1]
    torch.testing.assert_close(actual, inputs * 2)


@pytest.mark.parametrize("failure", ["missing", "extra", "scalar", "nonfinite"])
def test_invalid_extractor_output_is_rejected_before_any_vector_is_cached(tmp_path, failure):
    audio = tmp_path / "audio.wav"
    audio.touch()
    frame = pd.DataFrame({"filename": [audio.name] * 2, "path": [str(audio)] * 2, "start": [0, 5]})

    def extract(batch):
        match failure:
            case "missing":
                return torch.ones(1, 3)
            case "extra":
                return torch.ones(3, 3)
            case "scalar":
                return torch.ones(2)
            case _:
                return torch.full((2, 3), float("nan"))

    with pytest.raises(ValueError, match="one finite feature vector"):
        embeddings(
            frame,
            FeatureDataset(torch.ones(2, 4)),
            extract,
            loader=LoaderConfig(predict_batch_size=2),
            cache_dir=tmp_path / "cache",
        )
    assert not list(tmp_path.rglob("*.npy"))
