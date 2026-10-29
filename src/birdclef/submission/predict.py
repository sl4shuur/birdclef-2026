"""CPU inference from saved runs, with optional probability averaging."""

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from birdclef.competition import ROW_ID, class_columns, soundscape_windows
from birdclef.submission.build import build_submission
from birdclef.submission.write import write_submission


def predict_runs(
    runs: Sequence[Path],
    windows: pd.DataFrame,
    template: pd.DataFrame,
    *,
    weights: Sequence[float] | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Align IDs/classes and reuse each Perch encoder's features within this call.

    Dataset and notebook outputs use the same saved-run format. Inference always
    uses one CPU, FP32, no tracker and no pretrained-weight downloads.
    """
    from birdclef.config import LoaderConfig, RuntimeConfig
    from birdclef.data import InferenceSoundscapeDataset
    from birdclef.data.embeddings import embeddings
    from birdclef.data.sequences import recording_windows, requested_indices
    from birdclef.models.classifier.backends import PerchBackend
    from birdclef.models.runs import model_class, read_model_config

    if not runs:
        raise ValueError("Select at least one model run.")
    values = np.asarray(weights if weights is not None else [1.0] * len(runs), dtype=float)
    if (
        values.shape != (len(runs),)
        or not np.isfinite(values).all()
        or (values <= 0).any()
        or not np.isfinite(values.sum())
    ):
        raise ValueError("Provide one finite, positive weight per run.")
    loader = LoaderConfig(predict_batch_size=batch_size, num_workers=num_workers)
    classes = class_columns(template)
    predictions = []
    shared_features = {}
    for run in runs:
        print(f"Model: {run}", flush=True)
        config = read_model_config(run / "model-config.json")
        model = model_class(config.model_type).from_run(run)
        model.config.runtime = RuntimeConfig(accelerator="cpu", devices=1, precision="32-true")
        model.config.loader = loader
        backend = getattr(model, "backend", None)
        if isinstance(backend, PerchBackend) and len(runs) > 1:
            # Both Perch backends use the same fixed waveform preprocessing.
            # Canonical full recordings also provide the recurrent heads' context.
            context = recording_windows(windows)
            key = backend.encoder.sha256
            if key not in shared_features:
                shared_features[key] = embeddings(
                    context, InferenceSoundscapeDataset(context), backend.encoder, loader=loader
                )
            dataset = backend.prediction_data(context, features=shared_features[key])
            prediction = model.predict(context, dataset=dataset)
            prediction = prediction.iloc[requested_indices(windows, context).numpy()].copy()
            prediction[ROW_ID] = windows[ROW_ID].to_numpy()
        else:
            prediction = model.predict(windows)
        prediction = build_submission(prediction, template)
        predictions.append(prediction[classes].to_numpy())
    result = template.copy()
    result[classes] = np.average(predictions, axis=0, weights=values)
    return result


def run_inference(
    runs: Sequence[Path],
    competition_dir: Path,
    output: Path,
    *,
    weights: Sequence[float] | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    soundscapes: Path | None = None,
) -> Path:
    """Predict actual test files; public Run All checks one training minute instead.

    Kaggle exposes test audio only during scoring. The public preview saves real
    smoke predictions separately and copies the sample to submission.csv.
    """
    started = perf_counter()
    sample = pd.read_csv(competition_dir / "sample_submission.csv")
    classes = class_columns(sample)
    directory = soundscapes if soundscapes is not None else competition_dir / "test_soundscapes"
    preview = soundscapes is None and not any(directory.glob("*.ogg"))
    if preview:
        print("No visible test audio: checking one training minute before submission.", flush=True)
        directory = competition_dir / "train_soundscapes"
    windows = soundscape_windows(directory)
    if preview:
        windows = windows.loc[windows.filename.eq(windows.filename.iloc[0])]
    template = windows[[ROW_ID]].reindex(columns=[ROW_ID, *classes], fill_value=0.0)
    predictions = predict_runs(
        runs, windows, template, weights=weights, batch_size=batch_size, num_workers=num_workers
    )
    path = write_submission(
        predictions, output.with_name("smoke-predictions.csv") if preview else output, template
    )
    print(
        f"Predicted {len(windows)} windows in {perf_counter() - started:.1f}s: {path}", flush=True
    )
    if preview:
        path = write_submission(sample, output, sample)
        print("Public preview only: submission.csv contains sample rows; scoring uses test audio.")
    return path
