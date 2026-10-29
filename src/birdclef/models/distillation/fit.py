"""Feature pretraining from the existing Perch cache, without changing inference."""

import json
from pathlib import Path

import lightning as L
import pandas as pd
import rich_click as click
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import Dataset

from birdclef.cli_common import CONTEXT_SETTINGS
from birdclef.data import SourceWeightedRandomSampler, TrainingDataModule, load_audio_split
from birdclef.data.distillation import DistillationDataset
from birdclef.data.embeddings import embeddings
from birdclef.data.prepared import audio_training_datasets
from birdclef.data.sampling import MixedSourceDataset
from birdclef.experiment import RunContext
from birdclef.loggers import CustomLogger, apply_logging_config, wandb_run
from birdclef.models.classifier.backends import EfficientNetBackend
from birdclef.models.distillation.config import DistillationConfig
from birdclef.models.distillation.network import FeatureDistillationModule
from birdclef.models.encoders.perch import PerchEncoder
from birdclef.models.trainer import check_fit_inputs, fit_module
from birdclef.runtime import apply_environment_defaults


def distillation_data(
    config: DistillationConfig,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    classes: list[str],
    teacher: PerchEncoder,
) -> TrainingDataModule:
    # Existing split checks reject recording overlap; focal crops are centered.
    audio, validation = audio_training_datasets(
        train, valid, classes=classes, transform=None, random_crop=False
    )
    cache = config.data.cache_dir / "perch-center-v1" / teacher.sha256

    def paired(frame: pd.DataFrame, dataset: Dataset) -> DistillationDataset:
        features = embeddings(
            frame, dataset, teacher, loader=config.loader, cache_dir=cache, cache_only=True
        )
        return DistillationDataset(dataset, features)

    training = MixedSourceDataset(
        {
            source: paired(train.loc[train.source.eq(source)], dataset)
            for source, dataset in audio.by_source.items()
        }
    )
    validation = paired(valid, validation)
    sampler = SourceWeightedRandomSampler(
        training.sources,
        soundscape_probability=config.loader.soundscape_probability,
        generator=torch.Generator().manual_seed(config.loader.seed),
    )
    return TrainingDataModule(training, validation, config=config.loader, sampler=sampler)


def distill(
    config: DistillationConfig,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    classes: list[str],
    context: RunContext,
) -> None:
    L.seed_everything(config.train.seed, workers=True)
    teacher = PerchEncoder(config.teacher_path)
    data = distillation_data(config, train, valid, classes, teacher)
    columns = ["source", "filename", "path", "start", "end", "teacher_sha256"]
    check_fit_inputs(
        config,
        train.assign(teacher_sha256=teacher.sha256),
        valid.assign(teacher_sha256=teacher.sha256),
        columns,
        context,
    )
    continuing = (context.output_dir / "checkpoints/last.ckpt").is_file()
    backend = EfficientNetBackend(config, classes)
    encoder = backend.network_encoder(initialize_pretrained=not continuing)
    assert isinstance(data.valid, DistillationDataset)
    teacher_dim = data.valid.teacher_features.shape[1]
    module = FeatureDistillationModule(
        encoder, encoder.output_dim, teacher_dim, config.model, config.lr_scheduler
    )
    context.tracker.log_hyperparams({**config.safe_dump(), "teacher_dim": teacher_dim})
    best = fit_module(
        module,
        data,
        train_config=config.train,
        loader_config=config.loader,
        runtime=config.runtime,
        run=context,
    )
    if rank_zero_only.rank == 0:
        state = torch.load(best, map_location="cpu", weights_only=False)["state_dict"]
        encoder.load_state_dict(
            {
                key.removeprefix("encoder."): value
                for key, value in state.items()
                if key.startswith("encoder.")
            }
        )
        torch.save(encoder.cpu().state_dict(), context.output_dir / "encoder.pt")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--config", "config_path", required=True, type=click.Path(path_type=Path, exists=True)
)
@click.option("--name", required=True)
@click.option("--prepared-dir", type=click.Path(path_type=Path, exists=True, file_okay=False))
def fit(config_path: Path, name: str, prepared_dir: Path | None) -> None:
    """Train/resume Stage 1 and save encoder.pt from the best cosine checkpoint."""
    apply_environment_defaults()
    config = DistillationConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    apply_logging_config(config.logging)
    prepared_dir = prepared_dir or config.data.output_dir / "data_prep"
    classes = json.loads((prepared_dir / "classes.json").read_text(encoding="utf-8"))
    train, valid = load_audio_split(
        prepared_dir, config.data.competition_dir, fold=config.fold, classes=classes
    )
    if config.smoke:
        train = train.groupby("source", sort=False).head(32).reset_index(drop=True)
    run_name = config.run_name(name)
    output = config.data.output_dir / run_name
    with wandb_run(config.tracking, name=run_name, save_dir=output) as tracker:
        distill(
            config,
            train,
            valid,
            classes,
            RunContext(
                output,
                tracker,
                CustomLogger("distillation", config=config.logging),
                f"file-gkf5/fold-{config.fold}" + ("/smoke" if config.smoke else ""),
            ),
        )
    click.echo(f"Saved student encoder: {output / 'encoder.pt'}")


if __name__ == "__main__":
    fit()
