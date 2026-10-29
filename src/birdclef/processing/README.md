# Audio processing

`log_mel` turns a mono 32 kHz waveform into a Log-Mel input.

| Setting | Value |
| --- | --- |
| FFT / hop | 1024 / 320 samples |
| Mel bands | 128 |
| Frequency range | 50–14,000 Hz |
| dB reference / range | Power relative to 1 / 80 dB |
| Center padding | Off |
| Shape for 5 seconds | `(1, 128, 497)` |

This CPU transform has no learned statistics or augmentation. It does not pass
gradients. File loading belongs in `data`; learned transforms belong in a model.
Fit any learned data processing on training data only. Save it with the model.
