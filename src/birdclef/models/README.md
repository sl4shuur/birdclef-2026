# Models

`BirdClefModel` defines the `fit/predict` contract. Shared Lightning loss and
optimization live in `common.py`; `trainer.py` and `checkpoints.py` handle training
and checkpoint state.

[encoders/](encoders/README.md) contains reusable feature extractors and their
settings. Encoders know nothing about classes, targets or experiment lifecycles.

[classifier/](classifier/README.md) implements one-window classification:
configuration, encoder integration, heads, training and inference.
Public imports come from `birdclef.models`; optional ML dependencies load lazily.
