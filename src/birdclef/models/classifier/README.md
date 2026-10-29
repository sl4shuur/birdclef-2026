# Audio classifiers

`BirdClefModel` defines the shared `fit/predict` contract. `AudioClassifier` is the
classification implementation; different encoders use the same training lifecycle.

| File or folder | Responsibility |
| --- | --- |
| `model.py` | Shared training, prediction and validation reports |
| `network.py` | Encoder/head PyTorch module and classification metrics |
| `backends.py` | Connect raw audio or cached embeddings to training and saved runs |

```bash
python -m birdclef.models.fit --config <config.json> --name <run/name>
```

Public imports come from `birdclef.models`. Configs can load without optional ML
dependencies. Shared settings and CLI live in `models/config.py` and `models/fit.py`.
The lifecycle is reused by the recurrent model; its backend owns sequence assembly
and its prediction hook maps sequence outputs back to requested windows.
Reusable encoders live in [encoders/](../encoders/README.md), without
class labels, targets or run state. The classifier backend owns their integration:
datasets, feature dimensions and external artifacts. This is needed because
EfficientNet trains through PyTorch while frozen Perch feeds precomputed features.

`AudioClassifierModule` composes a PyTorch encoder and a head. It owns freezing,
partial/gradual unfreezing, optimizer groups and window-level validation. Perch's
head uses `nn.Identity` because feature extraction happens before PyTorch training.

To add an encoder, implement its feature extraction in `models/encoders/`, add an
`EncoderName` entry, then connect it through `backends.py`. New PyTorch encoders use
their direct child modules as fine-tuning blocks. No encoder names appear in the
shared lifecycle or exporter. `backend=` permits custom Python experiments;
the mapping is needed only for config-driven construction and saved-run loading.

See [EfficientNet](../encoders/efficientnet.md) and [Perch](../encoders/perch.md)
for model settings and notebook workflows.
