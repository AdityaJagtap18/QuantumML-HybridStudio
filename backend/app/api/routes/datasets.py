from fastapi import APIRouter, HTTPException

from app.schemas.dataset import DatasetConfigRequest, DatasetInfo, DatasetPreviewResponse
from app.services import image_dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetInfo])
def list_datasets() -> list[DatasetInfo]:
    return image_dataset_service.list_datasets()


@router.post("/preview", response_model=DatasetPreviewResponse)
def preview_dataset(config: DatasetConfigRequest) -> DatasetPreviewResponse:
    try:
        return image_dataset_service.build_preview(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
