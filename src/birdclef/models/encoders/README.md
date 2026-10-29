# Audio encoders

Encoders turn model inputs into `(batch, D)` feature vectors. They do not receive
class labels, targets, experiment configs or `RunContext`.

| File | Responsibility |
| --- | --- |
| `config.py` | `EncoderName` and shared encoder settings |
| `efficientnet.py` | Trainable PyTorch EfficientNet, normalization and pooling |
| `perch.py` | Frozen ONNX Perch and local model checksum |
| `download.py` | Explicit download of the pinned Perch weights |

`EfficientNetEncoder.output_dim` comes from the backbone. Perch's dimension is read
from its extracted or cached vectors. The classifier sizes its head accordingly.
No 1280/1536 feature dimension or class count is hardcoded in head construction.

[Classifier backends](../classifier/backends.py) own data, class order, training
and saved-run integration. [data/embeddings.py](../../data/embeddings.py) extracts
and caches features through any supplied callable, without constructing models.
Trainable encoders run inside the PyTorch network; their changing features are
never stored in a persistent training cache.

See [EfficientNet training](efficientnet.md) and [Perch + MLP](perch.md).
