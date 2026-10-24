# BirdCLEF+ 2026

Sources: [overview and evaluation](https://www.kaggle.com/competitions/birdclef-2026/overview), [data](https://www.kaggle.com/competitions/birdclef-2026/data), [organizer sample notebook](https://www.kaggle.com/code/stefankahl/birdclef-2026-sample-submission). Checked 2026-10-24.

## Task and files

Multilabel identification of wildlife in Pantanal soundscapes, with 234 species/sonotype targets across birds and other animal groups. Take class names and order from `sample_submission.csv`; join taxonomy on `primary_label`.

- `train.csv` and `train_audio/`: focal recordings with weak primary/secondary clip-level annotations.
- `train_soundscapes/` and `train_soundscapes_labels.csv`: soundscapes and available temporal annotations.
- `test_soundscapes/`: hidden recordings made available for scoring, in OGG at 32 kHz, 60 seconds per official file.
- `taxonomy.csv`: species metadata.
- `sample_submission.csv`: schema and placeholder rows, not a complete hidden-test manifest.

Each prediction covers five seconds. IDs use `<filename_stem>_<end_second>`: the 15–20 second interval ends in `_20`. Each minute produces twelve rows.

## Evaluation

Maximize macro ROC-AUC after excluding classes without positive targets. Per-class AUC compares positive and negative rankings, giving ties half credit. Class scores receive equal weight. Compute over the entire validation set.

The local `score` follows the organizer notebook and calls scikit-learn directly, without Kaggle utilities. It expects aligned rows and does not mutate inputs. No-positive classes are excluded; an entirely negative validation set raises ValueError. Other undefined-AUC behavior is inherited from the installed scikit-learn version. Inspect split coverage.

## Inference requirements

Submit via a Kaggle notebook, writing `submission.csv`. Scoring runs on CPU within 90 minutes, without Internet. Attach dependencies and weights as inputs. Public external datasets and pretrained weights are allowed under competition rules; respect their licenses and availability.

The final deadline was June 3, 2026. Late submission availability is controlled by Kaggle.

## Validation

The repository supplies an unstratified recording-level holdout and a split manifest. Keep windows of each recording together. Use site-level grouping when assessing unseen-site generalization. A final model-selection protocol has not been chosen.

Focal recordings differ from passive soundscapes in conditions and annotation granularity. Rare classes may be absent locally. Correlated windows, site overlap and repeated tuning on a small labeled subset can inflate validation results. Record the protocol with scores; local and leaderboard scores need not agree.
