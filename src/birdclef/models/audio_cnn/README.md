# Acoustic baseline

`AudioCNN` implements the same `BirdClefModel.fit/predict` contract as the numeric
MLP. `AudioCNNModule` owns learned computation, BCE loss and full-epoch validation.
It inherits AdamW/scheduler handling from `BirdClefModule`, and uses the shared
trainer/checkpoint/resume guard. Training supports one device; validation metrics
and source sampling are not designed for distributed training yet.

The encoder is torchvision's [EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html),
optionally initialized from `IMAGENET1K_V1`. Its first RGB convolution is replaced
by a mono convolution initialized with the RGB weight mean. It accepts
`(B, 1, 128, 497)` float32 Log-Mel inputs at the standard five-second duration.
Each spectrogram is standardized by its own mean/std (std clamped to 1e-6);
no statistics are fitted on validation. No image resize, crop or RGB transforms
are applied. Global average pooling produces 1280 features; the dropout/linear
head outputs one independent logit per saved species, with sigmoid at inference.
All encoder and head parameters are trainable, using one learning rate.

`data.prepared.load_audio_split` verifies version-2 prepared artifact hashes and
class order, rebuilds audio paths and uses the saved fold assignments. Training
combines focal clips and annotated soundscape windows. Validation is only
soundscapes, with no shared soundscape filenames. Focal supervision remains weak;
this does not establish site independence or eliminate acoustic duplicates.

Macro ROC-AUC is calculated after concatenating the entire validation epoch, using
`eval.metrics.evaluate`, which preserves the competition metric's checks. Classes
without positives are reported with undefined per-class AUC and skipped in the
macro. A class without negatives raises before training. Sanity-check batches are
not treated as a full fold. The notebook monitors `val/auc` in max mode.

`fit` saves best/last checkpoints, model config, input signature, best-checkpoint
validation predictions and per-class support/AUC. Only max_epochs and checkpoint
count can change when resuming a run. Optimizer, scheduler and epoch state are
restored; sampling/crop RNG continuation is not guaranteed bit for bit. Changed
data, model or loader settings require a new run directory. The classifier order
and frontend are fixed by the saved config and this implementation.

`AudioCNN.from_run` creates the architecture with no external initialization;
`predict` loads all weights from the checkpoint. Inference needs neither targets
nor Internet and preserves manifest row IDs/order. Build test windows from actual
files with `competition.soundscape_windows`; sample submission IDs are placeholders.
Use the normal submission helpers to align/validate/write predictions.

Start `notebooks/local/audio_baseline.ipynb` with `SMOKE_RUN=True`. The training
subset and output directory differ from a full-data run; full validation remains
unchanged. Local CSV logging avoids external logging services. The CPU benchmark
uses one real minute and includes checkpoint loading, decode, Log-Mel and model
prediction. It is not a hidden-test runtime guarantee or a model-quality result.

The named `encoder`, `pool` and `head` provide replacement boundaries for future
freezing, learning-rate and head ablations. No such experiment policies are
implemented before the first full baseline measurement.
