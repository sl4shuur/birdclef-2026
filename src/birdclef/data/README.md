# Data

Implement loading competition files, creating validation splits and assembling
datasets and data loaders here. Add a Lightning DataModule only if needed.

Keep split membership reproducible and prevent leakage between training and
validation. Reuse sample transformations from `processing/`; keep model training
and metric calculation in their own modules.
