"""Small test inputs do not need large CPU thread pools."""

import os

import pytest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"


@pytest.fixture
def encoder_path(tmp_path):
    """Tiny real ONNX graph: two features, to catch hardcoded Perch dimensions."""
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["inputs", "axes"], ["mean"], keepdims=1),
            helper.make_node("Neg", ["mean"], ["negative"]),
            helper.make_node("Concat", ["mean", "negative"], ["embedding"], axis=1),
        ],
        "test-encoder",
        [helper.make_tensor_value_info("inputs", onnx.TensorProto.FLOAT, ["batch", 160000])],
        [helper.make_tensor_value_info("embedding", onnx.TensorProto.FLOAT, ["batch", 2])],
        [helper.make_tensor("axes", onnx.TensorProto.INT64, [1], [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    path = tmp_path / "test.onnx"
    onnx.save(model, path)
    return path


@pytest.fixture
def audio_frames(tmp_path):
    import numpy as np
    import pandas as pd

    sf = pytest.importorskip("soundfile")
    frames = []
    for name, source in [("focal", "focal"), ("train", "soundscape"), ("valid", "soundscape")]:
        path = tmp_path / f"{name}.wav"
        values = np.repeat(np.array([0.1, -0.2], dtype=np.float32), 5 * 32000)
        sf.write(path, values if source == "focal" else np.tile(values, 6), 32000)
        frames.append(
            pd.DataFrame(
                {
                    "filename": [path.name] * 2,
                    "path": [str(path)] * 2,
                    "source": [source] * 2,
                    "start": [0, 5],
                    "end": [5, 10],
                    "row_id": [f"{name}_5", f"{name}_10"],
                    "a": [1, 0],
                    "b": [0, 1],
                }
            )
        )
    focal, train, valid = frames
    return pd.concat([focal.iloc[:1], train], ignore_index=True), valid
