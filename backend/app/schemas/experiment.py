from pydantic import BaseModel

from app.schemas.architecture import ArchitectureConfig, PreprocessingConfig
from app.schemas.dataset import DatasetConfigRequest
from app.schemas.training import TrainingConfig


class ExperimentCreateRequest(BaseModel):
    name: str
    dataset: DatasetConfigRequest
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    architecture: ArchitectureConfig
    training: TrainingConfig = TrainingConfig()


class ExperimentSummary(BaseModel):
    id: str
    name: str
    status: str
    stage: str
    dataset_id: str
    architecture_kind: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None


class ExperimentDetail(ExperimentSummary):
    config: dict
    result: dict | None
    training_history: list | None
