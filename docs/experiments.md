# Results

Scores below come from the user's Kaggle submission history.
Higher ROC-AUC is better. Local validation and leaderboard scores are different.

## Training comparison (ablation table)

| Version | Training | Public score | Private score |
| --- | --- | ---: | ---: |
| v1 | Full fine-tuning | **0.80213** | **0.82236** |
| v2 | Partial unfreezing | 0.75442 | 0.75728 |
| v3 | Gradual unfreezing | 0.78110 | 0.80442 |

These are separate runs, not a controlled ablation. Some runs were stopped early.
Training length and checkpoints can differ. The table does not show the effect
of freezing alone. Full fine-tuning has the highest score in both columns here.
Linear probing has no submitted score in this record.

For new results, record the run name, fold, checkpoint, settings and local score.
Keep public and private scores in separate columns.
