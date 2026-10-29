# Inference and submission

Open `notebooks/kaggle/inference.ipynb`. Attach competition data, the
[wheelhouse](https://www.kaggle.com/datasets/denysmykhailov/birdclef-2026-wheelhouse),
and a model Dataset or a training notebook's saved Output.
The [model Dataset](https://www.kaggle.com/datasets/denysmykhailov/birdclef-2026-models)
has `full-finetune` and `partial-unfreeze`.
Choose CPU and Internet OFF. Installation uses `--no-index`.

Set `RUN_DIRS` to the model paths printed in the notebook. One run uses one model.
Several runs average probabilities after aligning IDs and classes.
`WEIGHTS=None` gives equal weights; otherwise give one positive weight per run.
Loading supports this project's AudioCNN runs. Other architectures need another loader.
Inference uses one CPU device, FP32, saved processing and class order. It starts no
W&B run and downloads no weights.

## Checkpoints and export

A run needs `model-config.json`, `classes.json` and a checkpoint.
`fit.json` selects the best checkpoint. If no best is recorded, use
`checkpoints/last.ckpt`. A missing recorded best is an error. Paths must stay inside the run.

```bash
uv run python -m birdclef.submission.export --run outputs/audio-cnn --run outputs/inference-models
```

The command searches subfolders. Overlapping paths are checked once. Use arrows,
Space and Enter to choose models; `a` toggles all. Ctrl+C cancels. An empty choice
writes nothing. Incomplete folders are skipped with a reason. Selected run folder
names must be unique.

The ZIP is `build/birdclef-models.zip`. Use `--output` for another path or `--all`
to skip the menu. It has config, classes, one checkpoint and an inference `fit.json`.
Original runs stay unchanged. Upload the ZIP as a private Kaggle Dataset.
Attaching saved notebook Output needs no export. Rebuild the wheelhouse after code changes.
The old full-tuning run has a converted copy in `outputs/inference-models/full-finetune`.

## Run from a terminal

```bash
python -m birdclef.submission --run <run-folder> --competition-dir <data-folder> --output submission.csv
```

Repeat `--run` for more models. Repeated `--weight` values follow their order.
Defaults: batch size 32, workers 0, CPU threads 4. Change them with
`--batch-size`, `--num-workers`, `--threads`.
`--soundscapes <folder>` runs on that audio and never writes sample rows.

## Test files and output

Hidden test audio appears only during scoring. If the public test folder is empty,
Run All checks one real training minute and saves `smoke-predictions.csv`.
It copies sample rows to `submission.csv`. When test audio is present, it predicts
all files and writes real scores. Missing folders or bad predictions cause an error.
IDs use filenames and 5-second end times, not the sample's placeholder IDs.

Both branches passed an offline Kaggle CPU check: two models, 12 windows, 234 classes.
This checks the code, not full test runtime or score quality.

`build_submission` aligns IDs/classes to a template. `validate_submission` checks
exact order, unique IDs and finite scores in [0, 1]. `write_submission` checks then
writes CSV without an index. No missing scores are filled or clipped.
