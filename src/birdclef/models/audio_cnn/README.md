# Acoustic baseline

`AudioCNN` implements the `BirdClefModel.fit/predict` contract.
It uses the shared trainer, checkpoint handling and resume guard. Script launches
support Lightning DDP on multiple GPUs, including gradual encoder unfreezing.

## Run locally

Install `uv sync --extra notebooks --extra audio --extra deep-learning`, select
the repository `.venv` kernel and run `notebooks/local/data_prep.ipynb`. Then open
`notebooks/local/audio_baseline.ipynb`, edit the parameters and run its `%run` cell.
Logs and progress appear directly in the notebook. Interrupt the cell to stop.
`SMOKE_RUN=True` runs one short epoch on a small mixed-source training subset and
retains the entire validation fold; `False` starts a separate full-data run.

The same entry point works from a terminal:

```bash
python -m birdclef.models.audio_cnn.fit --config <json> --name audio-cnn/linear-probing
```

Prepared artifacts default to `data.output_dir/data_prep`. The entry point owns
split loading, smoke sampling, W&B tracking, training and best-checkpoint reports.
Paths and tracking settings come from `.env` or the environment. `WANDB_API_KEY`
takes precedence over the configured key; saved configs exclude secrets. Offline
runs can be uploaded later with `wandb sync`.

## Run on Kaggle

Build `build/birdclef-kaggle.zip` with
`uv run --extra audio --extra deep-learning python -m scripts.kaggle.wheelhouse build`.
Attach it as a Dataset along with competition audio, import
`notebooks/kaggle/starter.ipynb`, and select GPU with Internet ON. The notebook
installs the fresh package, reads the saved folds/weights from the Dataset, and
runs the current local partial-unfreezing configuration through `!python -u -m`.
Use Save Version → Save & Run All; afterwards download `partial-unfreeze.zip`
from Output. W&B is offline by default and needs no secret.

The Kaggle starter targets T4 ×2: `devices=2`, FP16 mixed precision, 64 examples
per GPU (128 effective), two persistent data workers per GPU and pinned memory.
CPU/BLAS threads are limited to one per process to avoid oversubscription.
Convolutions use channels-last memory and cuDNN benchmarking. These settings
change the effective batch size; keep it fixed when comparing experiments.

A Kaggle T4 benchmark of the pretrained model after partial unfreezing measured
1302, 1228 and 1053 examples/s on one GPU at batches 64, 128 and 256 respectively.
This excludes audio I/O and Log-Mel preprocessing; it is not full training throughput.

## Model and validation

The encoder is torchvision's [EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html),
optionally initialized from `IMAGENET1K_V1`. Its RGB stem becomes a mono 3x3
convolution initialized with the RGB weight mean. Input is float32 Log-Mel with
shape `(B, 1, 128, 497)` for five seconds, standardized per spectrogram with a
minimum standard deviation of 1e-6. No statistics are fitted on validation.
Global average pooling yields 1280 features; the dropout/linear head produces
one logit per species. BCE with logits and AdamW train the selected parameters;
inference applies sigmoid. `encoder`, `pool` and `head` are explicit modules.

`load_audio_split` verifies version-2 prepared hashes and class order, rebuilds
paths and uses saved folds. Training combines focal clips and soundscape windows;
validation contains soundscapes with no filenames shared with training. Focal
supervision remains weak; this does not establish site independence or remove
acoustic duplicates.

Macro ROC-AUC uses `eval.score` on the complete validation epoch. Classes without
positives are skipped; a class without negatives raises before training. Sanity
batches are excluded. Detailed per-class AUC/support are calculated for the best
checkpoint after training. The notebook monitors `val/auc` in max mode.
DDP gathers validation shards, restores their sequential order and removes
sampler padding before computing AUC. Final predictions and artifacts are written
on rank zero using one GPU, preserving row order without distributed duplicates.

## Three fine-tuning experiments

Use the same fold, seed, loader, dropout, sampling probability and scheduler.
In the notebook set `SMOKE_RUN=False`; use `max_epochs=10` and
`early_stopping_patience=10` for all three runs. This gives the gradual experiment
two frozen epochs and eight fine-tuning epochs without early stopping cutting
either stage short. These learning rates are starting hypotheses, not measured
optima.

Change `RUN_NAME` and the fields in `AudioModelConfig`:

| Experiment | RUN_NAME | head_lr | encoder_lr | unfreeze_epoch | unfreeze_last_n |
| --- | --- | --- | --- | --- | --- |
| A: Linear probing | `audio-cnn/linear-probing` | `1e-3` | `3e-5` (unused) | `None` | `None` |
| B: Gradual unfreezing | `audio-cnn/gradual-unfreeze` | `3e-4` | `3e-5` | `2` | `None` |
| C: Partial unfreezing | `audio-cnn/partial-unfreeze` | `3e-4` | `3e-5` | `0` | `2` |

`unfreeze_epoch` is zero-based: `None` keeps the encoder frozen, `0` starts
fine-tuning immediately, and `2` unfreezes before the third training epoch.
`unfreeze_last_n=None` selects the entire encoder; `2` selects the last two of
its nine top-level torchvision feature blocks, including the final convolution.
Both encoder and head are registered in the optimizer from the beginning, even
when the encoder is frozen. Schedulers operate on both groups and resume their
saved state. Trainability is recomputed from the restored epoch.
DDP registers initially frozen parameters too, so gradients are synchronized
when those parameters become trainable in a later epoch.

Frozen blocks have gradients disabled and remain in eval mode when Lightning
switches the model back to training. Their BatchNorm running statistics and
stochastic depth remain fixed. The head stays trainable. The shared module owns
BCE/sigmoid and optimizer/scheduler creation; the acoustic module supplies the
two parameter groups and encoder trainability.

## Saved runs and inference

`fit` saves best/last checkpoints, config, input signature, `classes.json`,
`validation-predictions.csv`, `validation-class-metrics.csv` and
`validation-metrics.json`. W&B also receives the per-class table and a model
artifact containing the best checkpoint and these files with relative paths.
`wandb_run` preserves online identity and closes the run even after an error.

`RUN_NAME`/`--name` supplies a short relative path:

```text
outputs/audio-cnn/
  linear-probing/
  gradual-unfreeze/
  partial-unfreeze/
```

`config.run_name()` validates this path and adds `/smoke` for a smoke check.
Hyperparameters stay in the saved config. The shared resume guard checks ordered
column names, data hashes and training settings in `fit-inputs.json`; only epoch
budget and checkpoint count may change. Changed settings require a new name.
Optimizer, scheduler and epoch state resume; sampling/crop RNG continuation is
not guaranteed bit for bit.

Construct `AudioCNN(config, classes=classes)` using prepared class order. Classes
are dataset/model metadata, saved beside and inside the checkpoint and checked
at inference. The current format requires all three: config, ordered-column input
signature and checkpoint class metadata.

`AudioCNN.from_run` restores architecture without downloading pretrained weights;
`predict` loads the checkpoint and preserves manifest row IDs/order. It needs no
targets. Build windows from actual files with `competition.soundscape_windows`
and use the submission helpers to align and validate predictions. The notebook's
optional CPU benchmark predicts one real minute, including checkpoint loading,
decode, Log-Mel and inference. Its CSV and timing check local plumbing, not
model quality or hidden-test runtime.
