# Frozen Perch + MLP

Start with [perch_mlp.ipynb](../../../../../notebooks/local/perch_mlp.ipynb).
It uses the existing training CLI, prepared folds, source sampler, BCE, AdamW,
schedulers, AUC, early stopping, checkpoints, resume and W&B reports.
Distillation and recurrent layers are separate future experiments.

```bash
uv sync --extra notebooks --extra deep-learning --extra audio --extra perch
uv run python -m birdclef.models.classifier.perch.download
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

`hidden_dim=512` selects LayerNorm → Linear → ReLU → Dropout → Linear.
`hidden_dim=None` selects the existing dropout/linear head. Input size comes from
the embeddings; output size follows `classes.json`. Neither 1536 nor 234 is
hardcoded in the head. EfficientNet also supports `hidden_dim`, and its current
linear head and fine-tuning behavior remain the defaults.

`AudioClassifierModule` accepts an encoder and a head. `AudioCNNModule` adds
EfficientNet preprocessing and gradual/partial unfreezing. Frozen Perch runs
outside PyTorch and its head receives cached vectors through an identity encoder.
`AudioClassifier` owns the shared training and inference lifecycle. It delegates
encoder-specific work to `ClassifierEncoder`: `training_data`, `module`,
`prediction_data` and `restore`. `FrozenPerchEncoder` owns the cache, feature size
and ONNX artifacts; `EfficientNetEncoder` owns Log-Mel inputs and the CNN module.
The classifier, training CLI and exporter contain no checks for encoder names.

To add an encoder, subclass `ClassifierEncoder`, add an `EncoderName` enum member,
and map it to your class in `models/classifier/encoders.py`. The base already provides audio
datasets and resume checks. Override data preparation for cached features, and
set `artifacts` if the encoder needs external files. For a Python-only experiment,
you can pass an encoder instance directly with `AudioClassifier(..., encoder=...)`;
register it when you need saved-run loading. Display names live in `EncoderName`,
so changing the configured string does not change implementation branches.

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
| Kaggle | `/kaggle/working/cache` | Attach a complete cache Dataset and point to `/kaggle/input/<dataset>/cache` |

Each window is one float32 `.npy` entry under
`perch-center-v1/<encoder SHA-256>/`. Keys contain the source, filename, crop offset
and audio file size. They exclude machine paths, labels and folds, so changing the
head or fold reuses existing windows. Writes are atomic; interrupted extraction
continues from missing entries. A complete cache needs no writes or ONNX session.
An incomplete cache needs a writable folder.

Audio is assumed to be immutable competition data. A replacement with the same
filename and byte length is not detected: clear the cache after changing audio.
Change the cache namespace when changing decoding or crop policy. Each training
split loads its embedding matrix into RAM; audio waveforms are processed in batches.

For comparison with EfficientNet, keep folds, seeds and source sampling fixed.
Account for Perch's fixed focal crop versus EfficientNet's random training crops.
The notebook starts in smoke mode and still validates the full fold. First-time
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

For GPU caching and the first MLP on Colab, use
[the Colab notebook](../../../../../notebooks/colab/starter.ipynb). It stages audio
and cache on local VM storage and saves archives to the Shared Drive.
