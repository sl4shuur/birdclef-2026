# BirdCLEF 2026

A deep-learning project for the [BirdCLEF+ 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026), focused on identifying wildlife species from acoustic recordings.

![Angry Bird Red](https://media1.tenor.com/m/IZqylGLe1PwAAAAC/angry-bird-angry-bird-red.gif)

The task is **multi-label audio classification**: predict the presence of 234 animal classes in 5-second soundscape segments, evaluated using macro-averaged ROC-AUC excluding classes without positive targets.

The project supports **local, Kaggle, and Google Colab** workflows, with modular training, evaluation, and inference pipelines.

## TODO

### 1. Exploration & Baseline

- [x] Perform EDA: class distributions, audio characteristics, labels, sites, and data quality.
- [x] Explore additional publicly available audio datasets and pretrained models.
- [x] Design modular, readable, and scalable training and inference pipelines.
- [x] Add label encoding, recording-level holdout, competition metric, and submission tools.
- [x] Implement audio preprocessing and select a baseline validation protocol.
- [ ] Train a baseline model and implement CPU-compatible inference.
- [ ] Submit the baseline to Kaggle and record the leaderboard score.

### 2. Model Improvements

- [ ] Explore pretrained audio models, including Perch.
- [ ] Implement knowledge distillation from a pretrained teacher into a smaller student model.
- [ ] Introduce recurrent layers (e.g., GRU/LSTM) for temporal modeling.
- [ ] Experiment with other improvements, such as augmentations, sampling strategies, and pseudo-labeling.
- [ ] Evaluate improvements and submit the best solution.

### 3. Final Deliverables

- [ ] Reproducible Python training and inference code.
- [ ] Screenshots of Kaggle submission results.
- [ ] Short PDF report covering EDA, methodology, experiments, results, and limitations.
- [ ] Proposed future improvements with additional time and compute.

## Local Setup

### Prerequisites

- **Python 3.13** (see `.python-version`)
- **Git**
- **uv** — [installation guide](https://docs.astral.sh/uv/)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sl4shuur/birdclef-2026.git
   cd birdclef-2026
   ```

2. Install dependencies:

   ```bash
   uv sync --extra notebooks --extra deep-learning --extra audio
   ```

3. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Set `BCLEF_DATA__COMPETITION_DIR` to your local competition dataset path.

4. Verify the environment:

   ```bash
   uv run src/birdclef/config.py
   uv run --extra deep-learning scripts/check_runtime.py
   ```

5. Open a notebook from `notebooks/local/`, `notebooks/kaggle/`, or `notebooks/colab/`. For local execution, select the project's `.venv` kernel.

### Tests & Code Quality

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Project Structure

| Directory                  | Purpose                                 |
| -------------------------- | --------------------------------------- |
| `src/birdclef/data/`       | Data loading and validation splits      |
| `src/birdclef/processing/` | Audio preprocessing and transformations |
| `src/birdclef/models/`     | Model architectures and training        |
| `src/birdclef/eval/`       | Metrics and evaluation                  |
| `src/birdclef/submission/` | Inference and submission generation     |
| `src/birdclef/experiment/` | Experiment orchestration and artifacts  |
| `src/birdclef/runtime/`    | Runtime and hardware configuration      |
| `notebooks/`               | Local, Kaggle, and Colab workflows      |
| `scripts/`                 | Runtime checks and deployment utilities |

Configuration is managed through `src/birdclef/config.py` and `.env`.

The acoustic model in `models/audio_cnn/` uses EfficientNet-B0 with a mono Log-Mel
input, separate encoder/head modules and configurable partial/gradual unfreezing.

## Kaggle & Colab

The Kaggle starter installs dependencies with `%pip`, saves a config without API
keys and launches `!python -u -m birdclef.models.audio_cnn.fit`. The script owns
training, W&B and checkpoint resume.

### Kaggle

For the current partial-unfreezing experiment, first run local data preparation
and build the input bundle:

```bash
uv run --extra audio --extra deep-learning python -m scripts.kaggle.wheelhouse build
```

This creates `build/birdclef-kaggle.zip`: the current project wheel, portable
prepared data with the saved folds, and cached ImageNet weights. It downloads
the ImageNet weights once if they are missing locally. No source audio or API
keys are included. Rebuild the bundle after changing package code or prepared
data.

1. Upload the ZIP as a private Kaggle Dataset and attach it to a notebook together
   with the BirdCLEF+ 2026 competition data.
2. Import `notebooks/kaggle/starter.ipynb`. Select **GPU** and **Internet ON**;
   dependency installation uses the Internet while retaining compatible packages
   already installed in the Kaggle image.
3. Use **Save Version → Save & Run All**. This runs in a clean session; an
   interactive Run All followed by Save & Run All trains twice.
4. Download `partial-unfreeze.zip` from the saved version's Output. The complete
   run also remains in `/kaggle/working/audio-cnn/partial-unfreeze/`, including
   best/last checkpoints, config, class order, validation reports and offline
   W&B history.

The notebook uses fold 0, encoder/head LRs 3e-5/3e-4, unfreezing the last two
encoder blocks at epoch 1 (the second epoch), up to six epochs and early-stopping
patience 3. Select **GPU T4 ×2**: the starter uses DDP, FP16 and batch 64 per GPU
(128 effective), with two persistent data workers per GPU and pinned memory.
`WANDB_OFFLINE=True` requires no credentials. For online tracking, add the Kaggle
secret `WANDB_API_KEY`, allow this notebook to access it, and set the flag to
`False`. All run outputs are saved locally in either mode.

#### Offline wheelhouse

`publish` uploads the complete training bundle and Linux dependency wheels to the
private Dataset configured in `scripts/kaggle/dataset.py`. Configure Kaggle CLI
credentials first.

```bash
uv run python -m scripts.kaggle.wheelhouse snapshot
uv run python -m scripts.kaggle.wheelhouse publish --dry-run
uv run python -m scripts.kaggle.wheelhouse publish --create
```

Use `publish -m "Update wheelhouse"` for subsequent updates.

The current private bundle is [BirdCLEF 2026 run](https://www.kaggle.com/datasets/denysmykhailov/birdclef-2026-run).

The saved image snapshot determines the target Python version. Compatible image
packages are reused; outdated packages and their dependencies are upgraded in the
bundle. Refresh the snapshot when Kaggle changes its image; this uses GPU quota.
Attach the published Dataset to the starter notebook. Online W&B tracking also
requires Internet access.

The current image's `google-adk` pins older OpenTelemetry versions, so pip warns
when W&B 0.31 upgrades them. BirdCLEF does not use `google-adk`; bundle installation,
offline W&B logging and a model forward/backward pass were verified on Kaggle.

### Google Colab

```bash
uv run python -m scripts.colab.wheelhouse build
```

Upload `build/birdclef-colab.zip` and follow the setup cells in `notebooks/colab/starter.ipynb`.

## Competition Constraints

- **Metric:** Macro ROC-AUC
- **Prediction:** 234 scores per 5-second audio segment
- **Inference:** CPU-only submission, 90-minute runtime limit
- **Internet:** Disabled during submission
- **External data:** Publicly available data and pretrained models are permitted, subject to competition rules

The competition has concluded; leaderboard submissions depend on Kaggle's post-competition availability.

## Audio data preparation

`FocalAudioDataset` returns random 5-second training crops or centered evaluation crops with weak clip-level targets. `SoundscapeDataset` reads exact annotated windows, never adding unannotated negatives. Both return mono 32 kHz waveforms and binary targets in the supplied class order; an optional `log_mel` transform returns `(1, 128, 497)` tensors. See `src/birdclef/data/README.md` for contracts.

Run `notebooks/local/data_prep.ipynb` using the local starter environment. Paths
come from the config: `data.competition_dir` (default `data/competition`) and
`data.output_dir` (artifacts in `outputs/data_prep`). Focal clips shorter than one
second are excluded with a report; original audio files remain untouched. Target
CSVs include identity keys (`filename` / `row_id`) and align by key after reload.

The notebook visualizes the audio-to-tensor stages and source sampling.
`loader.soundscape_probability` is a training hyperparameter: 0.3 means an expected
30% soundscape windows under `SourceWeightedRandomSampler`, independently of
source sizes. Call `sampler.set_soundscape_probability(0.3)` between epochs to
change the mixture with the same loader and model. An active iterator and its
prefetched batches are unchanged. Pass the sampler to `TrainingDataModule`;
validation is never oversampled. This is source balancing, not species balancing.

## Acoustic baseline

1. Install the notebook and training dependencies:
   `uv sync --extra notebooks --extra audio --extra deep-learning`.
2. Select the repository `.venv` kernel and run `notebooks/local/data_prep.ipynb`.
3. Open `notebooks/local/audio_baseline.ipynb`, set parameters and run the training
   cell. It calls the shared CLI with `%run`, so logs and progress appear directly
   in the notebook. Interrupt the cell to stop training.

`SMOKE_RUN=True` uses a small mixed-source training subset for one epoch, retaining
full-fold validation. Set it to `False` for a full run in a separate directory.
The notebook also shows validation reports and an optional CPU inference check
with a validated sanity-check CSV. Smoke scores do not establish baseline quality.

The model uses a mono EfficientNet-B0 encoder and a dropout/linear head on
Log-Mel inputs with BCE and AdamW. Encoder freezing and encoder/head learning
rates are configurable. First use with `pretrained=True` downloads
ImageNet weights; saved checkpoints restore without Internet. See
[`models/audio_cnn/README.md`](src/birdclef/models/audio_cnn/README.md) for details.

### Configuration, tracking and resume

Paths and tracking project/entity come from `.env` or the environment. Set
`WANDB_API_KEY` in the environment or `BCLEF_TRACKING__API_KEY` in `.env` for online
logging; `WANDB_OFFLINE=True` in the notebook stores a local run for `wandb sync`.
Saved configs exclude API keys. W&B logs LR/loss/AUC, per-class results and a
best-model artifact containing the checkpoint, config and validation reports.
`wandb_run` closes the run on completion, interruption or error and preserves
online run identity in `wandb-run-id.txt`.

Both local training notebooks use `%run` with one device. The acoustic entry point
also runs directly from a terminal:

```bash
uv run --extra audio --extra deep-learning python -m birdclef.models.audio_cnn.fit --config outputs/configs/audio-cnn/linear-probing.json --name audio-cnn/linear-probing
```

`--name` is the required relative experiment path, for example
`audio-cnn/linear-probing`. Smoke checks use a `smoke/` subfolder. The notebook
starts with experiment A; the [model README](src/birdclef/models/audio_cnn/README.md)
lists settings for A (linear probing), B (gradual unfreezing) and C (partial
unfreezing). `--prepared-dir` overrides `data.output_dir/data_prep`.

Increasing `max_epochs` continues the same run. The shared input signature checks
data, ordered columns and training settings before resume; changing the freezing
schedule, learning rates or fold requires a separate run folder.

Species order comes from prepared `classes.json`, is saved with the run and
checkpoint, and is checked at inference. It is model metadata rather than a
hyperparameter; W&B config contains only the class count. Runs require the current
config, signature and checkpoint format.

### Public imports

Use package APIs such as `from birdclef.data import load_audio_split`,
`from birdclef.models import AudioCNN, AudioExperimentConfig, AudioModelConfig`
and `from birdclef.submission import build_submission, write_submission`.
Leaf-module imports remain supported. Optional ML exports load on demand, so
label utilities and model configs do not require Torch or librosa.

Pyright checks `src`, `scripts` and `tests` using the project `.venv`; dev
requirements include pandas-stubs for DataFrame types in Pyright/Pylance.

## Offline Kaggle submission

Use [`notebooks/kaggle/inference.ipynb`](notebooks/kaggle/inference.ipynb) with
**CPU** and **Internet OFF**. Attach the competition, the current
`birdclef-2026-wheelhouse` Dataset, and either a model Dataset or the saved Output
of your training notebook. Set `RUN_DIRS` to the printed model paths.

One run gives a baseline submission. Multiple runs average probabilities;
optional `WEIGHTS` controls their contributions. The pipeline uses the recorded
best checkpoint, or `last.ckpt` for an interrupted run without a recorded best,
with saved class order and preprocessing, forcing CPU / FP32 without W&B or
pretrained-weight downloads. In the public preview it tests one training minute;
actual scoring predicts all hidden test soundscapes.

For local models, pack a Dataset-ready ZIP:

```bash
uv run python -m birdclef.submission.export --run outputs/audio-cnn --run outputs/inference-models
```

The CLI searches recursively and shows model checkboxes (arrows, Space, Enter).
`--all` skips the prompt. This creates `build/birdclef-models.zip` with only selected
checkpoints and inference metadata; original runs are untouched.
The older baseline's converted copy is
`outputs/inference-models/full-finetune`; its original artifacts are preserved.
See [`submission/README.md`](src/birdclef/submission/README.md) for CLI inference,
weights, model format and the public-preview behavior.
