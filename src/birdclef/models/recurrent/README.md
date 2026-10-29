# Frozen Perch + recurrent classification

Start with [perch_recurrent.ipynb](../../../../notebooks/local/perch_recurrent.ipynb).
The first baseline is LayerNorm → one bidirectional GRU (128 units per direction)
→ dropout → linear head. Feature dimension and class count come from data.
`cell="lstm"`, `bidirectional=False` and `recurrent_layers` allow small ablations
without another training pipeline. There is no residual MLP branch in this baseline.

The encoder stays frozen. Each soundscape is a separate 12-window sequence;
focal clips use their existing centered embedding as a sequence of length 1.
Padding is added only when reading a sample, not stored for every focal clip.
Packed sequences prevent it from influencing the backward GRU/LSTM direction.

`data/sequences.py` expands only recordings present in the chosen split and sorts
their windows by time. Missing annotations remain context inputs, with `mask=False`.
BCE and validation AUC use labeled positions only. Validation gathers complete
sequences before removing DDP sampler padding and filtering unlabeled positions.
Prediction expands each requested recording to all 12 windows for context, then
returns only requested row IDs in their original order.

Folds remain recording-based. The source sampler now draws **sequences**: one focal
clip or one soundscape minute. The notebook uses `soundscape_probability=0.01`,
approximately 10% labeled windows for this dataset. Reusing MLP's value of 0.1 would
give soundscapes much more loss weight because each minute has multiple labels.
This is a starting point, not a measured optimum. Validation still has only 13
recordings and 149 labeled windows on fold 0; use more folds before drawing conclusions.

The existing cache is reused. Complete recording context needs the 53 previously
unannotated windows in the current prepared dataset; they are extracted once,
without assigning them negative targets. A previously published MLP cache alone
is incomplete for recurrent training: copy it to writable storage to fill misses,
then republish if it must be mounted read-only. DDP workers require a complete
cache as described in [Perch settings](../encoders/perch.md).

Training, resume, AUC reports, W&B and checkpoints use the shared lifecycle.
Set `SMOKE_RUN=True` for one epoch with a small training subset and full validation.
Smoke selection preserves all annotations of each selected soundscape recording.
Use another run name for full training or a new cell/configuration.

```bash
python -m birdclef.models.fit --config <config.json> --name perch/bigru
python -m birdclef.submission.export --run outputs/perch/bigru --all
```

The saved config records `model_type="recurrent"`. The existing Kaggle inference
notebook selects the matching model automatically and supports ensembles with
window models. Export contains the recurrent checkpoint, class order and Perch
ONNX weights; CPU inference needs neither internet nor a training cache.
