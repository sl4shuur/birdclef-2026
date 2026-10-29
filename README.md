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

## Prepare and train

1. Run [data_prep.ipynb](notebooks/local/data_prep.ipynb). It saves files in `outputs/data_prep`.
2. Train a model. Like here for example: [audio_baseline.ipynb](notebooks/local/audio_baseline.ipynb).
3. Set a run name and training settings. Use `SMOKE_RUN=True` for a short check.
4. Results and checkpoints go to `outputs/<run name>`.

Run the same name to continue training. Use a new name when you change the data, fold or training settings. Saved configs do not include API keys.

For Kaggle or Colab, see [remote setup](docs/remote.md).
For model settings and saved files, see [training details](src/birdclef/models/audio_cnn/README.md).

## Submit on Kaggle

Open [inference.ipynb](notebooks/kaggle/inference.ipynb).
Attach competition data, the [wheelhouse](https://www.kaggle.com/datasets/denysmykhailov/birdclef-2026-wheelhouse), and your model Dataset or a training notebook's saved Output.
Choose **CPU** and **Internet OFF**. Set `RUN_DIRS` and run all cells. One run uses one model; several runs average their scores. `WEIGHTS` is optional.

To upload local models, first make a ZIP:

```bash
uv run python -m birdclef.submission.export --run outputs/audio-cnn
```

Choose models with Space, then press Enter. The ZIP is `build/birdclef-models.zip`.
You can repeat `--run` for more folders. Add `--all` to skip the menu.
See [submission details](src/birdclef/submission/README.md) for checkpoint rules and test data.

## More information

- [Results](docs/experiments.md)
- [Competition and metric](docs/competition.md)
- [Package layout](src/birdclef/README.md)
- [Data preparation](src/birdclef/data/README.md)
