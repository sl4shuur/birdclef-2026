# Cached Perch head experiments — 2026-10-29

All candidates are independent single checkpoints. No ensembles, weight averaging or TTA were used.

## Protocol

Frozen Perch features, original prepared recording folds and class order. All cache misses fail
instead of triggering ONNX extraction. Training used one RTX 5050 Laptop GPU, FP32, four CPU
threads, at most 30 epochs and early-stopping patience 6. AdamW and existing training/inference code
were reused. Original runs in `outputs/perch/{mlp,bigru,bilstm}` were preserved.

Fold 0 evaluates 149 labeled windows / 28 positive classes; fold 1 evaluates 147 / 34.
Unannotated recurrent context is excluded from loss and AUC. The same focal data is reused,
while the validation soundscape recordings are excluded from each fold's training set.

The original MLP used auto/mixed precision and two loader workers. Search runs use FP32 and zero
workers. Its matched control scores 0.92870, so compare hyperparameters against that control
rather than attributing the full difference from the earlier 0.92631 to the new parameters.

## Fold-0 screen, seed 42

Unspecified settings match the original family baseline. RNNs use one bidirectional layer,
128 units per direction, dropout 0.3, LR 0.001, batch 128, soundscape sequence probability 0.01.
MLPs use hidden 512, dropout 0.3, LR 0.001, batch 256, soundscape window probability 0.1.
Weight decay is 0.01 unless changed below. Baseline controls use the same search epoch budget.

| Candidate | Model changes | Local AUC | Best epoch (zero-based) | Epochs trained |
| --- | --- | ---: | ---: | ---: |
| mlp512-drop50 | MLP hidden=512, drop=0.5, LR=0.001, WD=0.01, pSC=0.1 | 0.93036 | 4 | 11 |
| mlp1024 | MLP hidden=1024, drop=0.5, LR=0.001, WD=0.01, pSC=0.1 | 0.93012 | 4 | 11 |
| mlp512-reg | MLP hidden=512, drop=0.4, LR=0.0005, WD=0.03, pSC=0.1 | 0.92937 | 10 | 17 |
| mlp-baseline | MLP hidden=512, drop=0.3, LR=0.001, WD=0.01, pSC=0.1 | 0.92870 | 4 | 11 |
| lstm256-reg | lstm, H=256, bidirectional, drop=0.5, LR=0.0005, WD=0.01, pSC=0.01 | 0.92474 | 18 | 25 |
| lstm256-uni | lstm, H=256, forward only, drop=0.3, LR=0.0005, WD=0.01, pSC=0.01 | 0.92109 | 12 | 19 |
| gru256 | gru, H=256, bidirectional, drop=0.3, LR=0.0005, WD=0.01, pSC=0.01 | 0.91855 | 6 | 13 |
| lstm256 | lstm, H=256, bidirectional, drop=0.3, LR=0.001, WD=0.01, pSC=0.01 | 0.91715 | 4 | 11 |
| lstm128-head512 | lstm, H=128, bidirectional, head=512, drop=0.3, LR=0.001, WD=0.01, pSC=0.01 | 0.91556 | 6 | 13 |
| lstm-baseline | lstm, H=128, bidirectional, drop=0.3, LR=0.001, WD=0.01, pSC=0.01 | 0.91371 | 16 | 23 |
| lstm128-sc03 | lstm, H=128, bidirectional, drop=0.3, LR=0.001, WD=0.01, pSC=0.03 | 0.91231 | 18 | 25 |
| linear | Linear head, drop=0.3, LR=0.001, WD=0.01, pSC=0.1 | 0.82850 | 24 | 30 |

## Seed and fold checks

Each fold is trained separately; this table compares configurations, not one model tested
on recordings it has already trained on. Fold-1 models are not blended into submission candidates.

| Configuration | Fold 0, seed 42 | Fold 0, seed 43 | Fold 1, seed 42 |
| --- | ---: | ---: | ---: |
| mlp-baseline | 0.92870 | 0.92755 | 0.89985 |
| mlp512-drop50 | 0.93036 | 0.92756 | 0.91187 |
| lstm-baseline | 0.91371 | — | 0.91652 |
| lstm256-reg | 0.92474 | 0.93260 | 0.91952 |

## Selection

- **Recurrent candidate:** BiLSTM-256, dropout 0.5, LR 0.0005, seed 43. The same configuration
  improved over the baseline LSTM at seed 42 on both folds: 0.92474 vs 0.91371 and 0.91952 vs 0.91652.
  Seed 43 reached 0.93260 on fold 0. This is the stronger recurrent candidate to submit first.
- **Static candidate:** MLP-512, dropout 0.5, seed 42. It matched/slightly exceeded the larger
  MLP-1024 with half the hidden size. It improved fold 1 by 0.01202 vs the matched MLP control,
  but the fold-0 gain disappeared at seed 43. The improvement is not consistently established.
- Increasing recurrent soundscape probability from 0.01 to 0.03 did not help on fold 0.
  A linear head was much weaker (0.82850). Larger recurrent state alone was insufficient;
  lower LR plus stronger dropout was the most useful recurrent combination tested.

These are exploratory local validation results selected from several trials. Neither candidate
has a new leaderboard score yet; local gains do not guarantee an improvement over the submitted MLP.

## Online repetitions

| Run | Local AUC | W&B |
| --- | ---: | --- |
| lstm256-reg-f0-s43-online | 0.93260 | [run](https://forge.coreweave.com/wandb/dmykhailov-kyiv-school-of-economics/birdclef-2026/runs/6c61c29b6e264070b5c98a2d3a291edf) |
| mlp512-drop50-f0-s42-online | 0.93036 | [run](https://forge.coreweave.com/wandb/dmykhailov-kyiv-school-of-economics/birdclef-2026/runs/d80e74d81033402583d0ff3f73342d97) |

## Reproduction and artifacts

Run from the project root with the existing baseline folders, prepared folds and complete cache:

```bash
python -m scripts.perch_experiments lstm256-reg mlp512-drop50
python -m scripts.perch_experiments lstm256-reg --seed 43
python -m scripts.perch_experiments lstm256-reg mlp512-drop50 --fold 1
python -m scripts.perch_experiments lstm256-reg --seed 43 --online
python -m scripts.perch_experiments mlp512-drop50 --online
```

Full configs, best/last checkpoints, validation predictions and per-class metrics are under
`outputs/perch/search-20261029/<candidate>-f<fold>-s<seed>[-online]/`.
`results.csv` contains all completed runs, including controls and repetitions.
Offline curves are saved as CSV; online repetitions have their own W&B run IDs.

Two separate submission exports are ready:

| Archive | Checkpoint | Size |
| --- | --- | ---: |
| `build/perch-best-bilstm.zip` | `lstm256-reg-f0-s43-online/checkpoints/epoch013.ckpt` | 81.9 MB |
| `build/perch-best-mlp.zip` | `mlp512-drop50-f0-s42-online/checkpoints/epoch004.ckpt` | 49.9 MB |

Each ZIP contains one checkpoint, its configuration, class order and the ONNX encoder.
Neither includes cached features, audio or API credentials. Export again with:

```bash
python -m birdclef.submission.export --run outputs/perch/search-20261029/lstm256-reg-f0-s43-online --all --output build/perch-best-bilstm.zip
python -m birdclef.submission.export --run outputs/perch/search-20261029/mlp512-drop50-f0-s42-online --all --output build/perch-best-mlp.zip
```

Both extracted exports passed CPU prediction on a complete 12-window recording from raw audio,
with training-cache access explicitly blocked. Predictions matched independently computed cached
CPU references within `3e-7`; full cached CPU validation reproduced AUC 0.93260 and 0.93036.
The BiLSTM's original GPU predictions differed by up to `2.61e-4` due to numerical precision;
disabling cuDNN TF32 reduced GPU/CPU differences below `5e-7`, without changing AUC.
These checks cover model restoration and inference, not a new Kaggle submission.
Detailed verification is saved in `outputs/perch/search-20261029/candidate-checks.json`.
