# Training

Run `notebooks/local/data_prep.ipynb`, then open
`notebooks/local/audio_baseline.ipynb` with the project `.venv` kernel.
Set `RUN_NAME` and the settings in the code cell. `SMOKE_RUN=True` runs one short
epoch with full-fold validation. Use a new folder for full training.

Local `%run` shows progress in the notebook. Use one device there. For DDP, run
the CLI as a separate process. See [remote setup](../../../../docs/remote.md).

```bash
python -m birdclef.models.audio_cnn.fit --config <config.json> --name <run/name>
```

`--prepared-dir` changes the default `data.output_dir/data_prep` folder.
Paths and tracking settings come from `.env` or the environment. Online W&B uses
`WANDB_API_KEY` (first choice) or `BCLEF_TRACKING__API_KEY`.
Offline logs can be uploaded later with `wandb sync`.

## Settings

The current model uses a mono EfficientNet-B0 and a dropout/linear head.
Log-Mel inputs have shape `(B, 1, 128, 497)`. Each input is standardized by its
own mean and standard deviation (minimum 1e-6). The RGB stem uses mean RGB weights.
Global average pooling gives 1280 features. Training uses BCE, AdamW and sigmoid
probabilities at inference. There are no fitted input statistics.

The first pretrained run can download ImageNet weights. Saved models need no download.

| Setting | Meaning |
| --- | --- |
| `encoder_lr`, `head_lr` | Learning rates for the two parts |
| `unfreeze_epoch=None` | Keep the encoder frozen |
| `unfreeze_epoch=0` | Train it from the first epoch |
| `unfreeze_epoch=2` | Start before the third epoch |
| `unfreeze_last_n=None` | Use all nine encoder blocks |
| `unfreeze_last_n=2` | Use the last two blocks |

Example settings (starting points, not measured best settings):

| Run | Head LR | Encoder LR | Unfreeze epoch | Last blocks |
| --- | --- | --- | --- | --- |
| Linear probing | 1e-3 | 3e-5 (unused) | None | None |
| Gradual unfreezing | 3e-4 | 3e-5 | 2 | None |
| Partial unfreezing | 3e-4 | 3e-5 | 0 | 2 |

For a fair comparison, keep fold, seed, batch size, dropout and source mix the
same. A useful starting setup is 10 epochs with early-stopping patience 10,
so the gradual run can finish its planned frozen and unfrozen stages.

Frozen blocks keep BatchNorm statistics fixed and stochastic depth off.
All parameters stay in the optimizer from the start. The scheduler runs for both
groups each epoch. Resume restores optimizer/scheduler state and the freeze stage.

DDP registers frozen parameters too, so later unfreezing keeps gradients in sync.
It gathers validation results, restores row order and removes sampler padding.
Only rank zero writes final reports and runs final prediction on one device.
GPU training uses channels-last inputs, mixed precision and cuDNN benchmarking.
Batch size and worker count are per device; keep effective batch size fixed for comparisons.
A T4 forward/backward check measured 1302/1228/1053 samples per second at batches
64/128/256. These numbers exclude audio loading and processing.

## Runs and results

Runs use folders such as `outputs/audio-cnn/my-run`; smoke checks use `smoke/` below it.
Names must be relative paths, with no empty parts, `.` or `..`.

Each run saves config, classes, input checks, best/last checkpoints and validation
CSV/JSON reports. W&B gets loss, LR, AUC, per-class scores and the best model artifact.
API keys are not saved. `wandb-run-id.txt` keeps the same online run ID on resume.
The W&B run closes on success, error or interruption.

Raise `max_epochs` to continue the same run. Only epoch count and checkpoint count
may change. New data, class order, fold or training settings need a new run folder.
Resume checks data hashes and settings; preprocessing-code changes are not checked
yet. Random crops and sampling may not continue bit for bit.

Validation uses a full soundscape fold, with no shared recording files in training.
The score skips classes without positives; classes without negatives cause an error.
This file split does not guarantee new-site performance or remove duplicate sounds.
Saved local scores are one-fold scores, not pooled OOF scores.

`AudioCNN.from_run` loads the saved config and classes. It uses the best checkpoint
from `fit.json`, or `checkpoints/last.ckpt` if no best is recorded. Class order is
checked against the checkpoint. `predict` keeps input row IDs and needs no targets.
See [submission](../../submission/README.md) for export and offline inference.
