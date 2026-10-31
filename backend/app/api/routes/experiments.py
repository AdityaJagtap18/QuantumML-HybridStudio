from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.models.experiment import ExperimentRecord
from app.schemas.experiment import ExperimentCreateRequest, ExperimentDetail, ExperimentSummary
from app.services import experiment_service

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _iso_utc(dt: datetime | None) -> str | None:
    """SQLite drops tzinfo on round-trip even for timezone=True columns, so
    naive datetimes here are always UTC -- make that explicit before
    serializing (see the equivalent fix in Experiments 1 and 2)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _to_summary(record: ExperimentRecord) -> ExperimentSummary:
    return ExperimentSummary(
        id=record.id,
        name=record.name,
        status=record.status,
        stage=record.stage,
        dataset_id=record.dataset_id,
        architecture_kind=record.architecture_kind,
        created_at=_iso_utc(record.created_at),
        started_at=_iso_utc(record.started_at),
        completed_at=_iso_utc(record.completed_at),
        error=record.error,
    )


def _to_detail(record: ExperimentRecord) -> ExperimentDetail:
    return ExperimentDetail(
        **_to_summary(record).model_dump(),
        config=record.config,
        result=record.result,
        training_history=record.training_history,
    )


@router.get("", response_model=list[ExperimentSummary])
def list_experiments() -> list[ExperimentSummary]:
    return [_to_summary(r) for r in experiment_service.list_experiments()]


@router.post("", response_model=ExperimentDetail)
def create_experiment(request: ExperimentCreateRequest, background_tasks: BackgroundTasks) -> ExperimentDetail:
    record = experiment_service.create_experiment(request)
    background_tasks.add_task(experiment_service.run_experiment, record.id)
    return _to_detail(record)


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: str) -> ExperimentDetail:
    record = experiment_service.get_experiment(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _to_detail(record)


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: str) -> dict[str, bool]:
    deleted = experiment_service.delete_experiment(experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"deleted": True}
