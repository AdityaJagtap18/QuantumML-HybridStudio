from typing import Literal

from pydantic import BaseModel, Field

DatasetId = Literal["digits", "mnist_subset", "fashion_mnist_subset"]


class DatasetInfo(BaseModel):
    id: DatasetId
    name: str
    description: str
    image_height: int
    image_width: int
    channels: int
    n_samples: int
    n_classes: int
    class_names: list[str]
    available: bool
    unavailable_reason: str | None = None


class DatasetConfigRequest(BaseModel):
    dataset_id: DatasetId
    selected_classes: list[int] | None = Field(
        default=None,
        description="Subset of class labels to keep, e.g. [0, 1] for binary digit classification.",
    )
    max_samples: int | None = Field(
        default=None,
        description="Cap total samples used (quantum simulation is expensive); None uses the full selected subset.",
    )
    test_size: float = Field(default=0.2, gt=0.0, lt=0.9)
    val_size: float = Field(default=0.2, gt=0.0, lt=0.9)
    random_seed: int = Field(default=42)


class ClassDistributionEntry(BaseModel):
    label: int
    class_name: str
    count: int


class ImageSample(BaseModel):
    index: int
    label: int
    class_name: str
    pixels: list[list[float]]  # row-major grid, raw intensity values


class DatasetPreviewResponse(BaseModel):
    dataset: DatasetInfo
    total_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    class_distribution: list[ClassDistributionEntry]
    sample_images: list[ImageSample]
    pixel_min: float
    pixel_max: float
    pixel_mean: float
