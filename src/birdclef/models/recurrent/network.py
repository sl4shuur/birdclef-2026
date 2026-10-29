"""Recurrent classification with packed focal clips and masked window supervision."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from birdclef.config import LRSchedulerConfig
from birdclef.models.classifier.network import AudioClassifierModule, classification_head
from birdclef.models.common import TrainingBatch
from birdclef.models.recurrent.config import RecurrentModelConfig


class RecurrentHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, config: RecurrentModelConfig) -> None:
        super().__init__()
        self.normalize = nn.LayerNorm(input_dim)
        cell = {"gru": nn.GRU, "lstm": nn.LSTM}[config.cell]
        self.rnn = cell(
            input_dim,
            config.recurrent_dim,
            num_layers=config.recurrent_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.recurrent_layers > 1 else 0,
        )
        output_dim = config.recurrent_dim * (2 if config.bidirectional else 1)
        self.classifier = classification_head(output_dim, num_classes, config)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        inputs = self.normalize(inputs)
        if lengths is None:
            values, _ = self.rnn(inputs)
        else:
            packed = pack_padded_sequence(
                inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            values, _ = self.rnn(packed)
            values, _ = pad_packed_sequence(values, batch_first=True, total_length=inputs.shape[1])
        return self.classifier(values)


class RecurrentClassifierModule(AudioClassifierModule):
    def __init__(
        self,
        input_dim: int,
        config: RecurrentModelConfig,
        lr_scheduler: LRSchedulerConfig,
        *,
        classes: list[str],
    ) -> None:
        super().__init__(
            nn.Identity(),
            RecurrentHead(input_dim, len(classes), config),
            config,
            lr_scheduler,
            classes=classes,
        )
        self.head: RecurrentHead

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(inputs, lengths)

    def loss(self, batch: TrainingBatch) -> torch.Tensor:
        inputs, targets, mask, lengths = batch
        loss = self.criterion(self(inputs, lengths)[mask], targets[mask])
        if self._trainer is not None and self.trainer.world_size > 1:
            # DDP averages gradients; normalize by labeled windows across all ranks.
            count = mask.sum()
            counts = self.all_gather(count)
            assert isinstance(counts, torch.Tensor)
            loss = loss * count * self.trainer.world_size / counts.sum()
        return loss

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> None:
        inputs, targets, mask, lengths = batch
        logits = self(inputs, lengths)
        loss = self.criterion(logits[mask], targets[mask])
        self.log("val/loss", loss, batch_size=int(mask.sum()), sync_dist=True, prog_bar=True)
        # Keep full sequences until DDP padding is removed, then exclude masked positions.
        targets = targets.detach().masked_fill(~mask.unsqueeze(-1), float("nan"))
        self._validation.append((targets, logits.sigmoid().detach().float()))
