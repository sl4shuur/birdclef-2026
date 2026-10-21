# Kaggle Starter

A notebook-first template for Kaggle competitions, with support for local, Kaggle and Colab deep-learning workflows. Implement the competition-specific data pipeline, model, evaluation and submission using the provided stubs.

## Prerequisites for Local Setup

- **Python 3.13** installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/). See `.python-version` file for the exact version.
- **Git** installed on your system. You can download it from the [official Git website](https://git-scm.com/downloads).
- **uv CLI** installed and on your `PATH` (see [Astral uv docs](https://docs.astral.sh/uv/)).

### How to install uv?

For all systems should work the following command:

```bash
pip install uv
```

See the [Astral uv docs](https://docs.astral.sh/uv/) for more installation options.

## How setup the project?

1. Clone the `kaggle-starter` branch (this template is maintained separately from `main`):

   ```bash
   git clone --branch kaggle-starter --single-branch https://github.com/sl4shuur/python-template.git my-competition
   cd my-competition
   ```

2. Sync the dependencies:

   ```bash
   uv sync --extra notebooks --extra deep-learning
   ```

3. Copy the `.env.example` file to `.env` and update the environment variables as needed:

   ```bash
   cp .env.example .env
   ```

4. Inspect the configuration and check the local training runtime:

   ```bash
   uv run src/birdclef/config.py
   uv run --extra deep-learning scripts/check_runtime.py
   ```

   View the wheelhouse command-line options:

   ```bash
   uv run python -m scripts.kaggle.wheelhouse --help
   uv run python -m scripts.colab.wheelhouse -h
   ```

   You can also activate the virtual environment and invoke the module directly:

   ```bash
   # (Windows)
   .venv\Scripts\Activate

   # (macOS/Linux)
   source .venv/bin/activate

   python src/birdclef/config.py
   ```

5. Open a starter notebook under `notebooks/local`, `notebooks/kaggle` or `notebooks/colab`. For local work, select the repository `.venv` kernel. Adapt the example cells and training workflow to your competition.

6. Run linting and tests:

   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run pytest
   ```

## Template structure

Application settings live in `src/birdclef/config.py`; `.env.example` lists their environment variables. In notebooks, call `apply_environment_defaults()` before the first `get_config()` and set `KS_DATA__COMPETITION_DIR` to your data path.

| Folder under `src/birdclef/` | Purpose |
| --- | --- |
| `core/` | Shared schemas and errors |
| `data/`, `processing/` | Loading, splits, datasets and sample transformations |
| `models/` | Competition-specific model and training logic |
| `eval/`, `submission/` | Metrics, prediction validation and submission format |
| `experiment/` | Run orchestration, artifacts and optional resume state |
| `runtime/`, `loggers/` | Platform defaults, hardware settings and logging |

`models/baseline/` contains a working Lightning regression example with checkpoints and resume; see [models/README.md](src/birdclef/models/README.md) for its layout and commands. Adapt the data, model and metrics for each competition. The `experiment/` functions remain orchestration templates.

## Kaggle and Colab packages

Run these commands from the project root. Both bundles include the `notebooks` and `deep-learning` extras.

For Kaggle, configure Kaggle CLI credentials and set `DATASET` in `scripts/kaggle/dataset.py` to your Dataset ID (`username/dataset-name`):

```bash
uv run python -m scripts.kaggle.wheelhouse snapshot
uv run python -m scripts.kaggle.wheelhouse publish --dry-run
uv run python -m scripts.kaggle.wheelhouse publish --create  # first upload
uv run python -m scripts.kaggle.wheelhouse publish -m "Update wheelhouse"
```

`snapshot` runs a private GPU script and uses GPU quota. It records the image's packages and Python version and refreshes dependency pins. Use the same image/Python version for training; refresh the snapshot when they change. Attach the published Dataset to your Kaggle notebook for offline installation; the image's PyTorch is reused.

For Colab:

```bash
uv run python -m scripts.colab.wheelhouse build
```

Upload `build/kaggle-starter-colab.zip` and use the installation cell in `notebooks/colab/starter.ipynb`. Colab installs dependencies online. The wheel includes only `src/birdclef`; `scripts/` stays in the repository.

## Pulling updates to existing project

1. Add this repository as a remote if you haven't already:

   ```bash
   git remote add template https://github.com/sl4shuur/python-template
   ```

2. Fetch the latest changes from the template's `kaggle-starter` branch:

   ```bash
   git fetch template kaggle-starter:refs/remotes/template/kaggle-starter
   ```

3. Merge the changes into your current competition branch:

   ```bash
   git merge template/kaggle-starter --allow-unrelated-histories
   ```

4. Resolve any merge conflicts if they arise, then commit the changes.
5. Push the updated competition branch to your remote repository:

   ```bash
   git push origin HEAD
   ```

6. Remove the template remote if you no longer need it:

   ```bash
   git remote remove template
   ```
