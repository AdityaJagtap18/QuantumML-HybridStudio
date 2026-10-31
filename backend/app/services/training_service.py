"""The real PyTorch training loop — classical and hybrid models share this
exact code path, so the only difference between a "classical" and
"hybrid" run is the model's architecture, never the training procedure.

Every value in the returned history is the actual measured loss/accuracy
for that epoch. Nothing here invents numbers.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from app.schemas.training import TrainingConfig


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    duration_seconds: float


@dataclass
class TrainingResult:
    history: list[EpochRecord] = field(default_factory=list)
    total_seconds: float = 0.0
    stopped_epoch: int = 0
    early_stopped: bool = False
    best_val_loss: float = float("inf")


def _make_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return torch.optim.SGD(model.parameters(), lr=config.learning_rate)


def _batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator):
    n = X.shape[0]
    perm = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        yield X[idx], y[idx]


def _predict_and_accuracy(logits: torch.Tensor, y: torch.Tensor, n_outputs: int) -> float:
    if n_outputs == 1:
        preds = (torch.sigmoid(logits) > 0.5).float()
        return float((preds == y).float().mean())
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean())


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_outputs: int,
    config: TrainingConfig,
    checkpoint_fn=None,
    on_epoch=None,
) -> TrainingResult:
    torch.manual_seed(config.random_seed)
    generator = torch.Generator().manual_seed(config.random_seed)

    optimizer = _make_optimizer(model, config)
    loss_fn = nn.BCEWithLogitsLoss() if n_outputs == 1 else nn.CrossEntropyLoss()

    result = TrainingResult()
    epochs_without_improvement = 0
    start_total = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        train_correct = 0
        for xb, yb in _batches(X_train, y_train, config.batch_size, generator):
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item() * xb.shape[0])
            train_correct += _predict_and_accuracy(logits, yb, n_outputs) * xb.shape[0]

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = float(loss_fn(val_logits, y_val))
            val_accuracy = _predict_and_accuracy(val_logits, y_val, n_outputs)

        train_loss = float(sum(train_losses) / X_train.shape[0])
        train_accuracy = float(train_correct / X_train.shape[0])
        duration = time.perf_counter() - epoch_start

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=config.learning_rate,
            duration_seconds=duration,
        )
        result.history.append(record)
        if on_epoch is not None:
            on_epoch(record)

        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            epochs_without_improvement = 0
            if checkpoint_fn is not None:
                checkpoint_fn()
        else:
            epochs_without_improvement += 1

        if config.early_stopping_patience and epochs_without_improvement >= config.early_stopping_patience:
            result.early_stopped = True
            result.stopped_epoch = epoch
            break
    else:
        result.stopped_epoch = config.epochs

    result.total_seconds = time.perf_counter() - start_total
    return result
