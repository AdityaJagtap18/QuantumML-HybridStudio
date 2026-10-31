import numpy as np

from app.schemas.dataset import DatasetConfigRequest
from app.services import image_dataset_service


def test_list_datasets_includes_digits_as_available():
    datasets = image_dataset_service.list_datasets()
    digits = next(d for d in datasets if d.id == "digits")
    assert digits.available is True
    others = [d for d in datasets if d.id != "digits"]
    assert all(not d.available for d in others)


def test_binary_digit_subset_has_two_classes():
    config = DatasetConfigRequest(dataset_id="digits", selected_classes=[0, 1])
    bundle = image_dataset_service.load_dataset(config)
    assert set(np.unique(bundle.labels)) == {0, 1}
    assert bundle.images.shape[1:] == (8, 8)


def test_max_samples_caps_dataset_size():
    config = DatasetConfigRequest(dataset_id="digits", selected_classes=[0, 1], max_samples=40)
    bundle = image_dataset_service.load_dataset(config)
    assert bundle.images.shape[0] == 40


def test_split_sizes_add_up():
    config = DatasetConfigRequest(
        dataset_id="digits", selected_classes=[0, 1], test_size=0.2, val_size=0.2, random_seed=0
    )
    bundle = image_dataset_service.load_dataset(config)
    X_train, X_val, X_test, y_train, y_val, y_test = image_dataset_service.split_dataset(bundle, config)
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert total == bundle.images.shape[0]
    assert X_train.shape[0] > X_val.shape[0]


def test_split_is_reproducible_for_same_seed():
    config = DatasetConfigRequest(dataset_id="digits", selected_classes=[0, 1], random_seed=7)
    bundle = image_dataset_service.load_dataset(config)
    split_a = image_dataset_service.split_dataset(bundle, config)
    split_b = image_dataset_service.split_dataset(bundle, config)
    for a, b in zip(split_a, split_b):
        assert np.array_equal(a, b)


def test_unavailable_dataset_raises_clear_error():
    config = DatasetConfigRequest(dataset_id="mnist_subset")
    try:
        image_dataset_service.load_dataset(config)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not wired up" in str(exc).lower() or "not available" in str(exc).lower()
