from typing import Literal

from pydantic import BaseModel, Field

OptimizerType = Literal["adam", "sgd"]


class TrainingConfig(BaseModel):
    epochs: int = Field(default=10, ge=1, le=200)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    optimizer: OptimizerType = "adam"
    random_seed: int = 42
    early_stopping_patience: int | None = Field(default=5, ge=1, le=50)
