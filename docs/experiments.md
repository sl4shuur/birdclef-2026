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
and **0.05494 private** ROC-AUC. It has the highest measured scores among our single-model submissions.
Training stopped early after 10 epochs. The encoder stayed frozen; only the head trained.

The head uses LayerNorm, a 512-unit hidden layer, ReLU, dropout 0.3 and an output layer for 234 classes. Head LR: `1e-3`; weight decay: `0.01`; batch size: `256`; soundscape sampling probability: `0.1`; seed: `42`.

## Perch v2: recurrent baselines

| Experiment | Kaggle notebook version | Local validation ROC-AUC | Public score | Private score |
| --- | --- | ---: | ---: | ---: |
| BiGRU, 128 units per direction | BirdCLEF submit, v5 | 0.90582 | 0.82664 | 0.83858 |
| BiLSTM, 128 units per direction | BirdCLEF submit, v6 | 0.91371 | 0.86567 | 0.88212 |

Both use one bidirectional layer, dropout 0.3, LR `1e-3`, batch size 128,
soundscape sequence probability 0.01 and seed 42.

## Cached head search

| Experiment | Kaggle notebook version | Local validation ROC-AUC | Public score | Private score |
| --- | --- | ---: | ---: | ---: |
| lstm256-reg: BiLSTM-256, dropout 0.5, LR `5e-4`, seed 43 | BirdCLEF submit, v7 | **0.93260** | 0.86839 | 0.87881 |
| mlp512-drop50: MLP-512, dropout 0.5, LR `1e-3`, seed 42 | BirdCLEF submit, v8 | 0.93036 | **0.88911** | **0.89546** |

Higher local AUC did not improve private scores. The original MLP remains the best single model
(public 0.89013, private 0.89707). [Full search report](perch-head-search-20261029.md).

The [4th-place solution writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/4-th-place-solution) reports **~0.928 standalone LB for Perch v2** in its model summary table.

## Perch -> EfficientNet distillation

| Experiment | Kaggle notebook version | Local ROC-AUC | Public | Private |
| --- | --- | ---: | ---: | ---: |
| ImageNet EfficientNet-B0, full fine-tuning (historical baseline) | BirdCLEF submit, v1 | 0.87771 | 0.80213 | 0.82236 |
| Perch feature distillation -> EfficientNet-B0 full fine-tuning | BirdCLEF submit, v16 | **0.89463** | **0.82217** | **0.84502** |

## Final Perch ensembles

| Experiment | Weights | Local ROC-AUC | Public | Private |
| --- | --- | ---: | ---: | ---: |
| E1 | MLP 50%, MLP dropout 0.5 50% | 0.92940 | 0.89161 | 0.89848 |
| E2 | MLP 80%, baseline BiLSTM 20% | 0.92633 | 0.89093 | **0.90087** |
| E3 | MLP 80%, BiLSTM-256 20% | 0.93161 | 0.89245 | 0.89913 |
| E4 | MLP 50%, MLP dropout 0.5 30%, BiLSTM-256 20% | 0.93330 | **0.89393** | 0.90016 |
| E5 | MLP 50%, BiLSTM-256 50% | 0.93408 | 0.89311 | 0.90051 |
