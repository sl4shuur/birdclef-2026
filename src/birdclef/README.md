# Package layout

| Folder | Use |
| --- | --- |
| `data` | Read audio, labels and saved folds |
| `processing` | Turn samples into model inputs |
| `models` | Train and load models |
| `eval` | Calculate scores |
| `submission` | Make and check submission files |
| `experiment` | Save run files and tracking logs |
| `runtime` | Set paths and choose hardware |

`competition.py` has shared file and audio rules. `config.py` has typed settings.
Models use the `BirdClefModel.fit/predict` interface.

Import public names from packages, for example `from birdclef.models import AudioClassifier`.
Model configs and label tools can load without Torch or librosa. ML code loads
when you use it. See each folder's README for details.
