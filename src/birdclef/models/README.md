# Models

`BirdClefModel` defines the `fit/predict` contract. Shared Lightning loss and
optimization live in `common.py`; `trainer.py` and `checkpoints.py` handle training
and checkpoint state.

[encoders/](encoders/README.md) contains reusable feature extractors and their
settings. Encoders know nothing about classes, targets or experiment lifecycles.

[classifier/](classifier/README.md) implements one-window classification and the
shared fit/reporting lifecycle. [recurrent/](recurrent/README.md) adds sequence
assembly and masked recurrent training, using the same encoders and lifecycle.
Common model settings live in `config.py`; `runs.py` selects the model family
for the shared `python -m birdclef.models.fit` CLI, export and inference.
Public imports come from `birdclef.models`; optional ML dependencies load lazily.

[distillation/](distillation/README.md) pretrains an EfficientNet encoder against
cached Perch features. Supervised fine-tuning and inference use the ordinary classifier.
