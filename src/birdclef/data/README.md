# Data

`labels.encode_labels` builds multi-hot targets for focal recordings in sample class order.
`split.group_holdout` holds out a fraction of unique recording groups, keeping all rows of each group together. It returns train/val frames and a manifest with the group column and keys.The helper is unstratified and takes explicit arguments, not automatic settings.
`birdclef.competition.soundscape_windows` creates official 5-second window manifests.

Audio decoding, transformations and temporal soundscape label preparation remain future work. Keep model training and metric calculation in their own modules. See the root README for usage.

`features.multilabel_dataset` prepares `(inputs, binary_targets)` tensors from precomputed features. `features.FeatureDataset` prepares inference inputs without requiring target columns. Datasets own row construction, loading, crop/target granularity and tensor conversion; models own learned computation, loss and predictions. Differentiable transforms may live inside a model. Clip-level weak labels must not be implicitly copied to every soundscape window.
