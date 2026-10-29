# Kaggle and Colab

## Saved Perch cache and model

The private [Perch cache Dataset](https://www.kaggle.com/datasets/denysmykhailov/birdclef-2026-perch-cache)
contains 35,918 verified float32 embeddings. Kaggle exposes the uploaded archive
under `perch-cache/`; set `data.cache_dir` to that folder, which contains
`perch-center-v1/`. The cache has no head weights, source audio or labels.

For Colab, upload `build/perch-cache.zip` to
`/content/drive/Shareddrives/BirdCLEF2026/data/cache/perch-cache.zip`.
Unpack it into the VM's cache folder, preserving `perch-center-v1/` and `index.json`.
The Colab starter is outdated; see the Colab section below.

For offline submission, export the saved Perch run:

```bash
uv run python -m birdclef.submission.export --run outputs/perch/mlp --all
```

Upload `build/birdclef-models.zip` to the existing models Dataset. Rebuild the
wheelhouse after code changes, including the `perch` extra for ONNX Runtime.
Use `notebooks/kaggle/inference.ipynb`, CPU and Internet OFF. Attach the
competition, models and wheelhouse only; no cache Dataset is needed.

## Perch embeddings

The [Perch notebook](../notebooks/local/perch_mlp.ipynb) also works after installing
the project bundle on Kaggle/Colab. Rebuild bundles to include ONNX Runtime.
Point `PREPARED_DIR` at the bundle's `data_prep`, and `ENCODER_PATH` at a downloaded
or attached Perch ONNX file. `CACHE_DIR` can be a writable working/Drive folder or
a complete read-only cache Dataset. Preserve the `perch-center-v1` directory when
uploading a cache. See [cache details](../src/birdclef/models/encoders/perch.md).
Perch extraction uses CPU by default; `model.encoder_device="cuda"` enables CUDA
with `onnxruntime-gpu`. Existing Kaggle
inference accepts the exported run, which includes the encoder weights.

## Kaggle training

Run local data preparation first. Then build the package and data bundle:

```bash
uv run python -m scripts.kaggle.wheelhouse build
```

The file `build/birdclef-kaggle.zip` has the package, saved folds and ImageNet
weights. It has no source audio or API keys. The first build can download weights.
Upload it as a private Dataset, or use the published wheelhouse.

Open `notebooks/kaggle/effnetb0-partial-unfreeze.ipynb` in Kaggle.
Attach competition data and the bundle. Choose GPU and Internet ON.
For two GPUs, choose T4 ×2. Batch size and worker count are per GPU.

**Add the Kaggle secret `WANDB_API_KEY` and allow the notebook to read it.**
`WANDB_OFFLINE=False` enables online tracking. Set it to `True` if you want local
W&B logs instead. You can upload those logs later with `wandb sync`.

Edit the settings in the notebook. Use Save Version -> Save & Run All for a clean
run. Running interactively first and then using Save & Run All trains twice.
Download the ZIP from Output when the run ends. The run folder also stays in
`/kaggle/working`.

## Publish a wheelhouse

Set up Kaggle CLI credentials. The Dataset ID is in `scripts/kaggle/dataset.py`.

```bash
uv run python -m scripts.kaggle.wheelhouse snapshot
uv run python -m scripts.kaggle.wheelhouse publish --dry-run
uv run python -m scripts.kaggle.wheelhouse publish --create
```

Use `--create` only for a new Dataset. For later updates:

```bash
uv run python -m scripts.kaggle.wheelhouse publish -m "Update package"
```

`publish` includes Linux wheels for missing or old packages. Rebuild after code
or prepared-data changes. Refresh the image snapshot when Kaggle changes its
image; the snapshot uses GPU quota. The target Python version comes from it.

W&B 0.31 updates OpenTelemetry. Kaggle may warn about `google-adk`, which uses
older versions. This project does not use `google-adk`.

## Colab

For distillation, use [perch_distillation.ipynb](../notebooks/colab/perch_distillation.ipynb).
Select a GPU runtime. Upload a fresh `birdclef-colab.zip` (package and `data_prep`),
`data/birdclef-audio.zip` (root folders `train_audio/` and `train_soundscapes/`),
and `data/cache/perch-cache.zip` to `/content/drive/Shareddrives/BirdCLEF2026/`.
The notebook copies archives to the VM and extracts them locally; checkpoints and
the final model ZIP are saved on Drive. Add the Colab secret `WANDB_API_KEY` for
online runs; configure `WANDB_BASE_URL` if using a custom W&B server.
Run smoke first, then set `SMOKE_RUN=False`. The teacher ONNX is downloaded automatically.

```bash
uv run python -m scripts.colab.wheelhouse build
```

The old `notebooks/colab/starter.ipynb` still
imports removed baseline classes and calls the removed baseline training CLI.
It is not a working training example until those cells are updated. Use the
current classifier API and CLI shown in the local notebooks when adapting it.
