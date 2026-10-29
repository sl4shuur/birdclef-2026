# Kaggle and Colab

## Perch embeddings

The [Perch notebook](../notebooks/local/perch_mlp.ipynb) also works after installing
the project bundle on Kaggle/Colab. Rebuild bundles to include ONNX Runtime.
Point `PREPARED_DIR` at the bundle's `data_prep`, and `ENCODER_PATH` at a downloaded
or attached Perch ONNX file. `CACHE_DIR` can be a writable working/Drive folder or
a complete read-only cache Dataset. Preserve the `perch-center-v1` directory when
uploading a cache. See [cache details](../src/birdclef/models/classifier/perch/README.md).
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

Edit the settings in the notebook. Use Save Version → Save & Run All for a clean
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

```bash
uv run python -m scripts.colab.wheelhouse build
```

Upload `build/birdclef-colab.zip` and follow `notebooks/colab/starter.ipynb`.
The starter still points to the removed baseline module. Update its training
command before use; it is not a working training example yet.
