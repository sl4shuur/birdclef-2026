# Audio classifiers

`BirdClefModel` defines the shared `fit/predict` contract. `AudioClassifier` is the
classification implementation; different encoders use the same training lifecycle.
There is no additional hierarchy of model types until another implementation needs it.

| File or folder | Responsibility |
| --- | --- |
| `config.py` | `EncoderName`, model and experiment settings |
| `model.py` | Shared training, prediction and validation reports |
| `network.py` | Encoder/head PyTorch module and classification metrics |
| `encoder.py` | Encoder interface and shared audio dataset preparation |
| `encoders.py` | Mapping from configured names to encoder implementations |
| `fit.py` | Training CLI for every supported encoder |
| `audio_cnn/` | EfficientNet implementation |
| `perch/` | Frozen Perch implementation, cache and download |

```bash
python -m birdclef.models.classifier.fit --config <config.json> --name <run/name>
```

Public imports come from `birdclef.models`. Configs can load without optional ML
dependencies. See [EfficientNet](audio_cnn/README.md) and [Perch](perch/README.md)
for model settings and notebook workflows.
