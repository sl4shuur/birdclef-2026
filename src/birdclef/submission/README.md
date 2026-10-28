# Submission

- `build.build_submission(predictions, template)` aligns explicit IDs and species to a template.
- `format.validate_submission(submission, template)` checks exact order, IDs and probabilities.
- `write.write_submission(submission, path, template)` validates, then writes the assembled CSV.

No clipping or filling. Hidden inference templates must use actual soundscape IDs; visible sample rows are placeholders. See the root README for an example.

## Offline inference

Open `notebooks/kaggle/inference.ipynb`. Attach the competition, the current
`birdclef-2026-wheelhouse` Dataset, and either a model Dataset or a training
notebook's saved Output. Choose CPU and Internet OFF. Installation uses
`--no-index`; restored models never download ImageNet weights or start W&B.

Ready inputs: [wheelhouse](https://www.kaggle.com/datasets/denysmykhailov/birdclef-2026-wheelhouse)
and [two baseline models](https://www.kaggle.com/datasets/denysmykhailov/birdclef-2026-models)
(`full-finetune`, `partial-unfreeze`).

Verified on Kaggle with Internet OFF, Torch 2.11.0+cpu, both models, 12 real
windows and all 234 classes. The public-preview and explicit-audio branches
produced identical predictions. This does not measure hidden-test quality or
full scoring runtime.

Set `RUN_DIRS` to explicit run folders. Both input sources use the same layout:

```text
partial-unfreeze/
  model-config.json
  classes.json
  fit.json
  checkpoints/epoch009.ckpt
```

`fit.json` selects the best checkpoint; its relative path must be preserved.
The notebook prints all available run paths. One run produces a single-model
submission. Several runs average probabilities after aligning row IDs and species;
`WEIGHTS=None` gives equal weights, or provide one positive weight per run.
Loading currently supports this project's `AudioCNN` runs; arbitrary external
architectures need their own model loader, not a filename convention.

The same pipeline runs locally or from a terminal:

```bash
python -m birdclef.submission --run outputs/audio-cnn/partial-unfreeze --competition-dir data/competition --output outputs/submission.csv
```

Repeat `--run` to ensemble. Optional repeated `--weight` values match run order.
Inference overrides the saved training runtime with CPU / one device / FP32;
`--batch-size`, `--num-workers` and `--threads` control execution. The defaults are
32, 0 and 4. Saved architecture and preprocessing remain unchanged.

Hidden test audio is [only visible during scoring](https://www.kaggle.com/competitions/birdclef-2026/discussion/681202).
When the public `test_soundscapes` directory is empty, Run All checks the selected
models on one real training minute, saves `smoke-predictions.csv`, and writes the
public sample rows to `submission.csv`. During scoring it predicts every actual
test file. A missing directory, bad model or invalid prediction raises an error.
For an explicit local audio folder use `--soundscapes <directory>`; this mode
always writes real predictions and never falls back to sample rows.

## Upload local models

Pack only config, classes, fit.json and the best checkpoint, omitting training
logs, last checkpoints and W&B history:

```bash
python -m birdclef.submission.export --run outputs/audio-cnn/partial-unfreeze --run outputs/inference-models/full-finetune --output build/birdclef-models.zip
```

Upload the ZIP as a private Kaggle Dataset, then attach it to the inference
notebook. Exported run folder names must be unique. Attaching a saved training
notebook's Output directly requires no export. Rebuild/publish the wheelhouse
after changes to the inference code; models and package dependencies are separate.

The existing older full-finetuning baseline has a converted copy in
`outputs/inference-models/full-finetune`; its original run is untouched. This was
a one-time config conversion, not a compatibility path in the inference code.
