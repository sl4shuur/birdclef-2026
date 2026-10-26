# BirdCLEF 2026

A deep-learning project for the [BirdCLEF+ 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026), focused on identifying wildlife species from acoustic recordings.

![Angry Bird Red](https://media1.tenor.com/m/IZqylGLe1PwAAAAC/angry-bird-angry-bird-red.gif)

The task is **multi-label audio classification**: predict the presence of 234 animal classes in 5-second soundscape segments, evaluated using macro-averaged ROC-AUC excluding classes without positive targets.

The project supports **local, Kaggle, and Google Colab** workflows, with modular training, evaluation, and inference pipelines.

## TODO

### 1. Exploration & Baseline

- [ ] Perform EDA: class distributions, audio characteristics, labels, sites, and data quality.
- [ ] Explore additional publicly available audio datasets and pretrained models.
- [ ] Design modular, readable, and scalable training and inference pipelines.
- [x] Add label encoding, recording-level holdout, competition metric, and submission tools.
- [ ] Implement audio preprocessing and select a baseline validation protocol.
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

`models/baseline/` contains a multilabel MLP for precomputed numeric features: BCE with logits for training, sigmoid probabilities for inference. Audio datasets and an optional Log-Mel frontend are available; an acoustic model remains future work. The generic experiment lifecycle files still contain explicit stubs; working MLP entry points are `birdclef.models.baseline.fit` and `birdclef.models.baseline.predict`.

## Kaggle & Colab

### Kaggle

For offline package installation, configure Kaggle CLI credentials and your dataset ID in `scripts/kaggle/dataset.py`.

```bash
uv run python -m scripts.kaggle.wheelhouse snapshot
uv run python -m scripts.kaggle.wheelhouse publish --dry-run
uv run python -m scripts.kaggle.wheelhouse publish --create
```

Use `publish -m "Update wheelhouse"` for subsequent updates.

The snapshot command uses Kaggle GPU quota. Attach the published wheelhouse dataset to the Kaggle notebook for offline installation.

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
