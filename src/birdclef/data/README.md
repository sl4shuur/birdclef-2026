# Data

`labels.encode_labels` owns species validation and focal multi-hot encoding in
sample-submission class order. Primary and optional secondary labels are weak
clip annotations, even when a training crop contains no target vocalization.

`manifests.focal_manifest` validates source paths and returns clip metadata and
aligned targets. `manifests.soundscape_manifest` returns only explicitly labeled,
aligned five-second windows and their targets. Duplicate annotation sets collapse;
conflicting sets, empty labels, unknown species and invalid intervals fail. Missing
windows are never negatives. Neither helper changes source files.

`audio.FocalAudioDataset` reads random training crops using torch RNG, centered
evaluation crops, and right-pads short clips. `audio.SoundscapeDataset` reads
exact annotated windows and rejects truncated audio. Both downmix, resample to
32 kHz and return `(inputs, binary_targets)`. Inputs are five-second waveforms,
or the result of an optional sample transform such as `processing.audio.log_mel`.
Targets must have a named, unique identity index: `filename` for focal clips,
`row_id` for soundscape windows. Read CSVs with `index_col` set to that key.
Datasets explicitly align by key and reject missing, extra or duplicate IDs.
Construct both sources with the same class order. No peak normalization is applied.

`audio.read_audio_window` also supports inference without targets. Build its
window metadata with `birdclef.competition.soundscape_windows`. Keep row IDs
aligned with predictions; the public sample contains placeholder rows.

`split.group_folds` creates zero-based, unstratified GroupKFold assignments,
stable under input row reordering. Save assignments before filtering windows.
`split.group_holdout` remains available for a single reproducible holdout, with
a manifest. Neither guarantees rare-class coverage. The preparation notebook
saves per-class positive/negative validation support: absence of positives skips
a class in the competition metric; absence of negatives blocks its AUC.

`notebooks/local/data_prep.ipynb` owns preparation artifact writing: portable CSV
metadata and targets, ordered `classes.json`, folds, support reports, header
audit and excluded short-recording review, plus source/artifact SHA-256 hashes. Paths are
rebuilt from a chosen data root; audio is not copied or content-hashed.

`features.multilabel_dataset` and `features.FeatureDataset` remain the entry points
for precomputed numeric features. Datasets own loading, crop/target granularity
and tensor conversion. Models own learned computation, loss and predictions;
differentiable transforms may instead live inside a model. The existing numeric
MLP does not accept waveforms or spectrograms.

`manifests.filter_focal_duration` applies the explicit minimum-duration policy
(default one second), returns retained metadata/targets and an exclusion report.
Exactly one second is retained; files are never deleted. The notebook runs the
header audit before filtering and saves version-2 artifacts with keyed targets.
Old target CSVs without keys must be regenerated.

`sampling.SourceWeightedRandomSampler` expects `ConcatDataset([focal, soundscapes])`.
It samples with replacement using weights `(1-p)/N_focal` and `p/N_soundscape`.
`p` is initialized from `loader.soundscape_probability` in the notebook. Update
with `set_soundscape_probability(p)` between epochs without rebuilding the model
or loader. Weights affect the next iterator; batches already drawn stay unchanged.
The fraction is expected, not guaranteed per batch. `num_samples` fixes epoch
length; no species balancing is implied. Pass the sampler explicitly to
`TrainingDataModule`. Use ordinary sequential loading for validation. Multi-device
distributed training requires a separate sampling design.
