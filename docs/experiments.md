# Results

Scores below come from the user's Kaggle submission history.
Higher ROC-AUC is better. Local validation and leaderboard scores are different.
Local validation uses `file-gkf5/fold-0`: 149 soundscape windows, with positives in 28 classes. Local values reflect the latest saved local results; historical Kaggle submissions may use earlier checkpoints.

## EfficientNet-B0 training comparison

| Version | Training           | Local validation ROC-AUC | Public score | Private score |
| ------- | ------------------ | -----------------------: | -----------: | ------------: |
| v1      | Full fine-tuning   |              **0.87771** |  **0.80213** |   **0.82236** |
| v2      | Partial unfreezing |                  0.85195 |      0.75442 |       0.75728 |
| v3      | Gradual unfreezing |                  0.86919 |      0.78110 |       0.80442 |
|         | Linear probing     |                  0.78959 |            - |             - |

These are separate runs, not a controlled ablation. Some runs were stopped early.
Training length and checkpoints can differ. The table does not show the effect of freezing alone. Full fine-tuning has the highest score in both columns here.

## Perch v2: one-window classification

| Experiment                                                              | Kaggle notebook version            | Local validation ROC-AUC | Public score | Private score |
| ----------------------------------------------------------------------- | ---------------------------------- | -----------------------: | -----------: | ------------: |
| Frozen Perch v2 + trained MLP head                                      | BirdCLEF submit, v4                |              **0.92631** |  **0.89013** |   **0.89707** |
| Perch v2 zero-shot with original classification head                    | BirdCLEF 2026: Native Perch v2, v1 |                        - |      0.82512 |       0.84213 |
| Perch v2 — 4th-place solution writeup standalone LB (split unspecified) | -                                  |                        - |       ~0.928 |             - |

The trained MLP improves on the native-head zero-shot submission by **0.06501 public**
and **0.05494 private** ROC-AUC. It has the highest measured scores among our submissions.
Training stopped early after 10 epochs. The encoder stayed frozen; only the head trained.

The head uses LayerNorm, a 512-unit hidden layer, ReLU, dropout 0.3 and an output layer for 234 classes. Head LR: `1e-3`; weight decay: `0.01`; batch size: `256`; soundscape sampling probability: `0.1`; seed: `42`.

The [4th-place solution writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/4-th-place-solution) reports **~0.928 standalone LB for Perch v2** in its model summary table.

For new results, record the run name, fold, checkpoint, settings and local score.
Keep public and private scores in separate columns.
