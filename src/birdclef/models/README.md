# Models

`BirdClefModel` defines the `fit/predict` contract. Shared Lightning loss and
optimization live in `common.py`; `trainer.py` and `checkpoints.py` handle training
and checkpoint state.

[classifier/](classifier/README.md) contains the audio classification family:
configuration, lifecycle, CLI, encoder interface, EfficientNet and frozen Perch.
Public imports come from `birdclef.models`; optional ML dependencies load lazily.
