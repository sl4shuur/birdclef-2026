# Perch → EfficientNet feature distillation

Start with `notebooks/local/perch_distillation.ipynb` (D1).
Use `SMOKE_RUN=True` for one epoch per stage; set it to `False` and Run All for
the real experiment. Smoke outputs live in separate `smoke/` folders.

1. **Features:** ImageNet EfficientNet-B0 → trainable projection → Perch features.
   Minimize mean cosine distance; train the entire encoder. The projection's dimensions
   come from the student and cached teacher vectors, not fixed constants.
   Use centered 5-second focal crops and exact soundscape windows, matching the existing
   `perch-center-v1/<teacher SHA256>` cache. Missing entries fail without extraction.
   Validation soundscape recordings only participate in validation. Select the best
   checkpoint by `val/loss`, then save only its encoder state as `encoder.pt`.
2. **Full fine-tuning:** initialize an ordinary `AudioClassifier` with
   `model.encoder_weights=<feature run>/encoder.pt` and a fresh 234-class linear head.
   Use BCE, normal random focal crops and `val/auc` checkpoint selection. Both learning
   rates are `3e-4` for D1. No projection or teacher is used at this stage.

Both notebooks use fold 0, seed 42, soundscape probability 0.1 and AdamW with weight
decay 0.01. The local example uses batch 32 and up to 5 epochs per stage;
the Colab example uses batch 64, 5 feature epochs and 10 supervised epochs.
The reported D1 run stopped after 3 supervised epochs.
Reduce batch size if GPU memory is insufficient. W&B is offline for smoke and online
for the real notebook run. Training resumes from each stage's `last.ckpt`.
The local notebook uses zero loader workers and disables pinned memory to reduce
Windows RAM/commit usage; increase workers only after measuring available memory.

The historical EfficientNet full-FT score is a reference (local AUC 0.87771).
For a controlled comparison, also train D0 with the same Stage-2 configuration,
fold, seed and epoch budget, setting `encoder_weights=None` in a separate run folder.
Smoke metrics only verify the pipeline; they do not measure distillation quality.

## Existing Kaggle submission

Export **only Stage 2** with the existing command:

```bash
python -m birdclef.submission.export --run outputs/distillation/d1/full-ft --all --output build/perch-distilled-efficientnet.zip
```

Update the models Dataset with the ZIP and the wheelhouse Dataset with this package:

```bash
python -m scripts.kaggle.wheelhouse publish
```

In `birdclef-submit`, select the exported student with `MODEL_WEIGHTS = {"full-ft": 1.0}`.
Do not attach Stage-1 training output: it also has
`fit.json` but is not a classifier. Export includes neither Perch, its cache,
`encoder.pt` nor the projection. CPU inference restores the classifier checkpoint
and does not read the original initialization file.
