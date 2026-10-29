"""Frozen Perch ONNX encoder. Its frontend consumes raw mono 32 kHz waveforms."""

from hashlib import file_digest
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from birdclef.competition import SAMPLE_RATE, SEGMENT_SECONDS


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


class PerchEncoder:
    def __init__(
        self, path: Path, *, threads: int = 4, device: Literal["cpu", "cuda"] = "cpu"
    ) -> None:
        self.path = path
        self.sha256 = file_sha256(path)
        self.threads = threads
        self.device = device
        self._session = None

    def _load(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        providers: list[str | tuple[str, dict[str, str]]] = ["CPUExecutionProvider"]
        if self.device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA extraction requires onnxruntime-gpu and an NVIDIA GPU.")
            ort.preload_dlls()
            # Avoid expensive cuDNN exhaustive searches when starting a short extraction run.
            providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
        session = ort.InferenceSession(str(self.path), sess_options=options, providers=providers)
        if self.device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("Perch CUDA provider failed to load; check CUDA/cuDNN installation.")
        inputs = session.get_inputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "inputs"
            or "embedding" not in {output.name for output in session.get_outputs()}
        ):
            raise ValueError("Expected a Perch ONNX model with inputs and embedding tensors.")
        return session

    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        values = waveforms.detach().cpu().numpy().astype(np.float32, copy=False)
        if values.ndim != 2 or values.shape[1] != SAMPLE_RATE * SEGMENT_SECONDS:
            raise ValueError("Perch expects waveforms shaped (batch, 160000).")
        if not np.isfinite(values).all():
            raise ValueError("Perch inputs must be finite.")
        if self._session is None:
            self._session = self._load()
        # Some exports fix the batch dimension to 1. Respect it rather than assuming batching.
        batch = self._session.get_inputs()[0].shape[0]
        size = batch if isinstance(batch, int) else len(values)
        outputs = []
        for start in range(0, len(values), size):
            chunk = values[start : start + size]
            count = len(chunk)
            if count < size:
                chunk = np.pad(chunk, ((0, size - count), (0, 0)))
            output = self._session.run(["embedding"], {"inputs": chunk})[0]
            outputs.append(np.asarray(output, dtype=np.float32)[:count])
        embeddings = np.concatenate(outputs)
        if embeddings.ndim != 2 or len(embeddings) != len(values):
            raise ValueError("Perch must return one embedding vector per waveform.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Perch returned nonfinite embeddings.")
        return torch.from_numpy(embeddings)
