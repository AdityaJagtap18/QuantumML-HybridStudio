"""Image dataset loading and splitting.

Only the digits dataset is wired up for now (Phase 1 scope, per the
implementation plan's "primary dataset first" rule) — MNIST/Fashion-MNIST
are listed in the catalog as not-yet-available rather than silently
missing, so the UI can say why instead of pretending they don't exist.

Preprocessing (normalization, flattening, PCA) happens in
preprocessing_service, strictly on the training split, to avoid leakage.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from app.schemas.dataset import (
    ClassDistributionEntry,
    DatasetConfigRequest,
    DatasetId,
    DatasetInfo,
    DatasetPreviewResponse,
    ImageSample,
)

_CATALOG: dict[DatasetId, DatasetInfo] = {
    "digits": DatasetInfo(
        id="digits",
        name="Digits (8x8 handwritten digits)",
        description="scikit-learn's built-in handwritten digit dataset — 8x8 grayscale "
        "images, 10 classes. Start with a binary subset (e.g. 0 vs 1) for the quantum layer.",
        image_height=8,
        image_width=8,
        channels=1,
        n_samples=1797,
        n_classes=10,
        class_names=[str(i) for i in range(10)],
        available=True,
    ),
    "mnist_subset": DatasetInfo(
        id="mnist_subset",
        name="MNIST (subset)",
        description="28x28 handwritten digits.",
        image_height=28,
        image_width=28,
        channels=1,
        n_samples=0,
        n_classes=10,
        class_names=[str(i) for i in range(10)],
        available=False,
        unavailable_reason="Not wired up yet — requires a torchvision download. "
        "Planned for a later phase; the digits dataset is the primary workflow.",
    ),
    "fashion_mnist_subset": DatasetInfo(
        id="fashion_mnist_subset",
        name="Fashion-MNIST (subset)",
        description="28x28 grayscale clothing images, 10 classes.",
        image_height=28,
        image_width=28,
        channels=1,
        n_samples=0,
        n_classes=10,
        class_names=[
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        available=False,
        unavailable_reason="Not wired up yet — requires a torchvision download. "
        "Planned for a later phase; the digits dataset is the primary workflow.",
    ),
}


@dataclass
class ImageDatasetBundle:
    info: DatasetInfo
    images: np.ndarray  # (n, h, w)
    labels: np.ndarray  # (n,)
    class_names: list[str]


def list_datasets() -> list[DatasetInfo]:
    return list(_CATALOG.values())


def get_dataset_info(dataset_id: DatasetId) -> DatasetInfo:
    if dataset_id not in _CATALOG:
        raise ValueError(f"Unknown dataset id: {dataset_id}")
    return _CATALOG[dataset_id]


def load_dataset(config: DatasetConfigRequest) -> ImageDatasetBundle:
    info = get_dataset_info(config.dataset_id)
    if not info.available:
        raise ValueError(info.unavailable_reason or f"{config.dataset_id} is not available yet")

    if config.dataset_id == "digits":
        raw = load_digits()
        images, labels = raw.images, raw.target
        class_names = [str(i) for i in range(10)]

        if config.selected_classes:
            mask = np.isin(labels, config.selected_classes)
            images, labels = images[mask], labels[mask]
            remap = {c: i for i, c in enumerate(sorted(config.selected_classes))}
            labels = np.array([remap[v] for v in labels])
            class_names = [str(c) for c in sorted(config.selected_classes)]

        if config.max_samples and images.shape[0] > config.max_samples:
            rng = np.random.default_rng(config.random_seed)
            idx = rng.choice(images.shape[0], size=config.max_samples, replace=False)
            images, labels = images[idx], labels[idx]

        return ImageDatasetBundle(info=info, images=images, labels=labels, class_names=class_names)

    raise ValueError(f"Unsupported dataset id: {config.dataset_id}")


def split_dataset(
    bundle: ImageDatasetBundle, config: DatasetConfigRequest
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X_train, X_val, X_test, y_train, y_val, y_test."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        bundle.images,
        bundle.labels,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=bundle.labels,
    )
    relative_val = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        random_state=config.random_seed,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preview(config: DatasetConfigRequest, max_preview_images: int = 24) -> DatasetPreviewResponse:
    bundle = load_dataset(config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(bundle, config)

    labels, counts = np.unique(bundle.labels, return_counts=True)
    class_distribution = [
        ClassDistributionEntry(label=int(label), class_name=bundle.class_names[int(label)], count=int(count))
        for label, count in zip(labels, counts)
    ]

    n_preview = min(max_preview_images, bundle.images.shape[0])
    rng = np.random.default_rng(config.random_seed)
    preview_idx = rng.choice(bundle.images.shape[0], size=n_preview, replace=False)
    sample_images = [
        ImageSample(
            index=int(i),
            label=int(bundle.labels[i]),
            class_name=bundle.class_names[int(bundle.labels[i])],
            pixels=bundle.images[i].tolist(),
        )
        for i in preview_idx
    ]

    info = bundle.info.model_copy(
        update={"n_samples": bundle.images.shape[0], "class_names": bundle.class_names}
    )

    return DatasetPreviewResponse(
        dataset=info,
        total_samples=bundle.images.shape[0],
        train_samples=X_train.shape[0],
        val_samples=X_val.shape[0],
        test_samples=X_test.shape[0],
        class_distribution=class_distribution,
        sample_images=sample_images,
        pixel_min=float(bundle.images.min()),
        pixel_max=float(bundle.images.max()),
        pixel_mean=float(bundle.images.mean()),
    )
