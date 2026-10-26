# Processing

Implement competition-specific transformations of individual samples here:
cleaning, normalization, tokenization, feature extraction or augmentation.

Keep transformations independent of file loading and training. Fit learned
preprocessing on training data only and reuse it for validation and inference.
Save any fitted preprocessing needed to reproduce predictions with the run artifacts.

`audio.log_mel` is the optional CPU baseline frontend: mono 32 kHz, FFT 1024,
hop 320, 128 Mel bands, 50-14,000 Hz, power dB re 1, 80 dB dynamic range,
no centering. A five-second waveform produces `(1, 128, 497)`. It contains no
learned statistics, augmentation or per-dataset normalization. It is not
differentiable; a learned waveform frontend belongs in the model instead.
