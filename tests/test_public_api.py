import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_lightweight_package_imports_do_not_load_optional_ml_dependencies():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from birdclef.data import encode_labels; "
            "from birdclef.models import AudioModelConfig; "
            "from birdclef.runtime import detect_environment; "
            "assert 'torch' not in sys.modules; assert 'librosa' not in sys.modules",
        ],
        env=env,
        check=True,
    )


def test_public_model_imports_resolve_without_circular_dependencies():
    pytest.importorskip("torchvision")
    from birdclef.data import SourceWeightedRandomSampler
    from birdclef.models import AudioClassifier, AudioCNNModule, BirdClefModule

    assert issubclass(AudioCNNModule, BirdClefModule)
    assert callable(AudioClassifier) and callable(SourceWeightedRandomSampler)
