# Data

## Prepare files

Run `notebooks/local/data_prep.ipynb`. It writes metadata, targets, class order,
fixed folds, support reports and SHA-256 hashes to `outputs/data_prep`.
Audio stays in its source folder. CSV paths are rebuilt from the chosen data root.
Hashes cover source tables and prepared files, not audio content.
Version-2 target CSVs need identity keys; rebuild older files without keys.

Class order comes from `sample_submission.csv`, not sorted taxonomy names.
Focal clips use primary and secondary labels for the whole clip. A short crop
may not contain the labeled sound. Soundscapes use only annotated 5-second windows.
Missing annotations are not negative examples.

Matching duplicate label sets are merged. Unknown species, bad intervals, empty
labels and conflicting label sets cause an error. Source files are not changed.
Focal clips below one second are excluded and listed in `short_recordings_review.csv`.
A clip of exactly one second is kept. Header read errors stop preparation.

## Load samples

- `FocalAudioDataset`: random training crops, centered evaluation crops; pad short clips.
- `SoundscapeDataset`: exact labeled windows; reject short or broken windows.
- `InferenceSoundscapeDataset`: the same windows without targets.

Audio becomes mono at 32 kHz. There is no peak normalization. Each sample is a
5-second waveform, or the result of an optional transform such as `log_mel`.
Targets are binary. Use the same class order for all datasets.

Target CSV indexes must be named `filename` for focal clips or `row_id` for windows.
Load them with `index_col`. Data is aligned by key; missing, extra or duplicate
keys cause an error. `soundscape_windows` makes inference IDs from real files.

## Folds and sampling

`group_folds` uses zero-based GroupKFold by recording. Save folds before filtering.
`group_holdout` gives one fixed split and a manifest. Neither ensures that rare
classes appear in every fold. Keep support reports. File groups do not prove
that sites or sounds are independent.

`load_audio_split` checks saved hashes and class order, rebuilds paths and uses
saved folds. Training has focal clips and soundscapes; validation has soundscapes
only. Shared soundscape files between train and validation cause an error.

`MixedSourceDataset` keeps named datasets and each example's source in concatenation
order. `SourceWeightedRandomSampler` takes that explicit source list, so reversing
the datasets cannot silently reverse their sampling probabilities.
It samples with replacement: weights are `(1-p)/N_focal` and `p/N_soundscape`.
`loader.soundscape_probability` sets `p` (default 0.1). This is an expected source
fraction, not a class balance rule or an exact fraction in every batch.
`num_samples` sets epoch length. Change `p` between epochs with
`set_soundscape_probability(p)`. Already drawn or prefetched samples stay the same.

Pass the sampler to `TrainingDataModule`. `set_epoch` uses seed + epoch, also
after resume. Lightning splits draws across DDP ranks. Validation is not sampled;
its scores use all unique windows after distributed padding is removed.

For saved numeric features, use `multilabel_dataset` and `FeatureDataset`.
Datasets handle loading and targets; models handle learned steps and predictions.

`embeddings.py` batches a supplied feature extractor and reads/writes a portable
cache. The caller chooses its complete namespace. It stores vectors and sample
identities, without labels, folds or model construction. A complete indexed cache
works without source audio and without writes; `cache_dir=None` extracts directly.
