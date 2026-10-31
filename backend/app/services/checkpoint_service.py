"""Save/restore a model's state_dict. Kept deliberately simple — one file
per experiment under artifacts/checkpoints/, named by experiment id."""

import torch
from torch import nn

from app.core.config import get_settings


def checkpoint_path(experiment_id: str) -> str:
    settings = get_settings()
    return str(settings.checkpoints_dir / f"{experiment_id}.pt")


def save_checkpoint(model: nn.Module, experiment_id: str) -> str:
    path = checkpoint_path(experiment_id)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model: nn.Module, path: str) -> None:
    model.load_state_dict(torch.load(path, map_location="cpu"))
