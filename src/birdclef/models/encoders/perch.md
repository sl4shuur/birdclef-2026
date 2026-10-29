# Frozen Perch + MLP

Start with [perch_mlp.ipynb](../../../../notebooks/local/perch_mlp.ipynb).
It uses the existing training CLI, prepared folds, source sampler, BCE, AdamW,
schedulers, AUC, early stopping, checkpoints, resume and W&B reports.
Distillation and recurrent layers are separate future experiments.

```bash
uv sync --extra notebooks --extra deep-learning --extra audio --extra perch
uv run python -m birdclef.models.encoders.download
```

The download command saves a pinned, checksum-verified ONNX backbone to
`data.cache_dir/models/perch_v2_backbone.onnx`. Training never downloads Perch
implicitly. An existing ONNX file can be supplied instead.

```python
AudioModelConfig(
    encoder=EncoderName.PERCH,
    encoder_path=Path("data/cache/models/perch_v2_backbone.onnx"),
    unfreeze_epoch=None,
    hidden_dim=512,
    dropout=0.3,
    head_lr=1e-3,
)
```

`hidden_dim=512` selects LayerNorm -> Linear -> ReLU -> Dropout -> Linear.
`hidden_dim=None` selects the existing dropout/linear head. Input size comes from
the embeddings; output size follows `classes.json`. Neither 1536 nor 234 is
hardcoded in the head. EfficientNet also supports `hidden_dim`, and its current
linear head and fine-tuning behavior remain the defaults.

`AudioClassifierModule` accepts an encoder and a head, and owns gradual/partial
unfreezing. `EfficientNetEncoder` owns normalization and pooling. Frozen Perch runs
outside PyTorch and its head receives cached vectors through an identity encoder.
`AudioClassifier` owns the shared training and inference lifecycle. It delegates
integration work to classifier backends: datasets, PyTorch modules and saved state.
`PerchBackend` owns cache selection, feature size and ONNX artifacts;
`EfficientNetBackend` selects Log-Mel inputs and constructs the trainable encoder.
The classifier, training CLI and exporter contain no checks for encoder names.

Encoders live independently in `models/encoders/`. Extend classifier integration
through `models/classifier/backends.py`; see the [classifier README](../classifier/README.md).
Shared feature extraction and caching live in `data/embeddings.py` and accept a
callable plus the cache namespace chosen by the backend. The cache has no Perch,
head, class or experiment dependencies.

## Audio and cache

The [ONNX backbone model card](https://huggingface.co/biodiversica/Perch-onnx-backbone)
specifies raw mono 32 kHz, five-second audio and an `embedding` output. Its frontend
is included. Existing audio reading handles downmixing and resampling; Log-Mel
preprocessing is used only for EfficientNet. ONNX Runtime defaults to CPU with four threads. Set `encoder_device="cuda"`
for GPU extraction with `onnxruntime-gpu`; exported inference still uses CPU.
Fixed and dynamic ONNX batch dimensions are supported.

Focal recordings use one deterministic centered crop, with right-padding for
short clips. Soundscapes use exact annotated windows. Labels remain weak for focal
clips. Validation uses the same saved recording-level folds as EfficientNet.
Labels/folds are read from prepared data, never from the embedding cache.

Set `BCLEF_DATA__CACHE_DIR` or `config.data.cache_dir`:

| Environment | Default | Reuse |
| --- | --- | --- |
| Local | `data/cache` | Normal folder |
| Colab | `/content/drive/Shareddrives/BirdCLEF2026/data/cache` | Mount Drive first |
| Kaggle | `/kaggle/working/cache` | Attach a complete cache Dataset and point to the folder containing `perch-center-v1/` |

Each window is one float32 `.npy` entry under
`perch-center-v1/<encoder SHA-256>/`. Keys contain the source, filename, crop offset
and audio file size. They exclude machine paths, labels and folds, so changing the
head or fold reuses existing windows. Writes are atomic; interrupted extraction
continues from missing entries. A complete cache needs no writes or ONNX session.
An incomplete cache needs a writable folder.

For the normal Lightning script launch, one parent process prepares the cache
before DDP workers start. Workers only read a complete indexed cache and reject
missing entries instead of extracting concurrently. With an external launcher
such as `torchrun`, prepare all training and validation embeddings in one process
first (for example, run the same config with `runtime.devices=1`). Use the same
cache directory in every worker. This does not coordinate separate cache-building jobs.

Each encoder namespace also stores `index.json`, mapping source/filename/crop to
the existing array filename. Once indexed, complete caches can be loaded without
the source audio. Existing arrays are indexed on first use while audio is available;
Perch is not rerun. Include `index.json` when copying or archiving the cache, and
initialize it before attaching a cache read-only.

Audio is assumed to be immutable competition data. After indexing, source audio
is not checked again, even if its byte length changes: clear the cache after replacing audio.
Change the cache namespace when changing decoding or crop policy. Each training
split loads its embedding matrix into RAM; audio waveforms are processed in batches.

For comparison with EfficientNet, keep folds, seeds and source sampling fixed.
Account for Perch's fixed focal crop versus EfficientNet's random training crops.
Set `SMOKE_RUN=True` for a short check that still validates the full fold. First-time
validation extraction can take several minutes. A smoke AUC is only a pipeline check.

## Offline inference and export

Each Perch run saves `encoder.onnx` and `encoder.json` (checksum and feature size)
alongside the existing artifacts. Export includes both, so the original training
paths and cache are not needed. `from_run` verifies the saved encoder checksum.
Inference decodes actual requested windows and extracts embeddings without writing
to a training cache. Existing inference and probability ensembles accept both kinds
of run. Rebuild the wheelhouse to include the new `perch` extra before offline use.

The notebook includes an uncached CPU minute check. Measure full inference in the
Kaggle environment before treating a local timing as a submission budget estimate.

The [Colab starter](../../../../notebooks/colab/starter.ipynb) still uses the removed
baseline API and is not a working Perch example. See [remote setup](../../../../docs/remote.md)
for its current status; adapt the local Perch notebook to the installed bundle.
